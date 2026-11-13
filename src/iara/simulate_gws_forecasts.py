''' Simulate GW mock catalogs from a detector-network detection forecast.

The forecast is a text file given by "detector_forecast" in the input yaml
(path relative to "storage", like the other paths, or an absolute path).
It is whitespace separated, with a header line naming the columns
(a leading '#' in the header is allowed; extra columns are ignored):

    z           true redshift of the source (optional if Dl is given)
    Dl          luminosity distance (optional if z is given)
    ra, dec     equatorial coordinates of the source [deg]
    err_Dl      1-sigma instrumental error on Dl (lensing is added separately)
    err_area90  90% credible sky area [deg^2]

Every event of the catalog inside the redshift bins (bin_edges) is simulated.
The hosts are drawn from the galaxy count maps {gdir}[_rand][_bias]/gal_count_map{k}.npy
written by simulate_galaxies.py. With include_lensing, the convergence is read from
{gdir}[_rand]/kappa_map{k}.npy, of shape (Nbins, npix), saved alongside them.

Input yaml parameters used by this module, besides the usual ones
(storage, bdir, gdir, det, nside, N_sim, N_samples, bin_edges, bgw,
include_lensing and the fiducial cosmology):

    "detector_forecast": "/forecasts/et2ce.txt"  # forecast catalog
    "forecast_distance_unit": "Mpc"     # unit of Dl and err_Dl: "Mpc" or "Gpc"
    "forecast_sky_selection": "ra_averaged"  # "ra_averaged", "full" or "none" (see gw_forecast._selection_maps)
    "forecast_sel_dz": 0.5              # thickness of the redshift bins of the selection maps
    "forecast_sel_nside": 8             # nside of the event histograms of the selection maps
    "forecast_sel_smoothing_deg": 10.0  # Gaussian smoothing of the selection maps [deg]
    "forecast_err_nside": 4             # max nside of the sky pixels used to assign errors
    "forecast_save_samples": False      # save the (Dl, lat, lon) samples of every event

All of them except "detector_forecast" are optional and default to the values above.
'''

import pandas as pd
import numpy as np
import healpy as hp
import camb
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
from scipy.optimize import fsolve
from scipy.stats import lognorm
import os, os.path


# If you want to parallelize
from mpi4py import MPI
from iara.simulate_galaxies import warn_existing_sims
from iara.defaults import with_defaults
rank = MPI.COMM_WORLD.Get_rank() # index of the task
size = MPI.COMM_WORLD.Get_size() # total number of tasks
barrier = MPI.COMM_WORLD.Barrier


def print_rank0(*args, **kwargs):
    """Print once (from rank 0), not once per MPI process."""
    if rank == 0:
        print(*args, **kwargs)


DELTAGAL=0.473477302
ALPHAGAL=3.366032942
LGAL= 0.576

def galactic2equatorial(lon, lat):
  """Transforms galactic coordinates lon, lat in equatorial"""
  sin_dgal=np.sin(DELTAGAL)
  cos_dgal=np.cos(DELTAGAL)
  ra= lon-LGAL
  sin_b=np.sin(lat)
  cos_b=np.cos(lat)
  sin_l=np.sin(ra)
  cos_l=np.cos(ra)

  sin_dec = cos_b*cos_dgal*sin_l + sin_b*sin_dgal
  sin_ra = cos_b*cos_l
  cos_ra = sin_b*cos_dgal - cos_b*sin_l*sin_dgal

  dec = np.arcsin(sin_dec)
  ra  = np.arctan2(sin_ra, cos_ra) + ALPHAGAL

  ra = ra % (2*np.pi)

  return ra, dec

def equatorial2galactic(ra, dec):
  sin_dgal = np.sin( DELTAGAL )
  cos_dgal = np.cos( DELTAGAL )
  tmp = -ALPHAGAL + ra

  sin_dec = np.sin(dec)
  cos_dec = np.cos(dec)
  sin_rag = np.sin(tmp)
  cos_rag = np.cos(tmp)

  sin_latg = cos_dec*cos_dgal*cos_rag + sin_dec*sin_dgal
  sin_long = sin_dec*cos_dgal - cos_dec*cos_rag*sin_dgal
  cos_long = cos_dec*sin_rag

  lat_g= np.arcsin( sin_latg )
  tmp = np.arctan2( sin_long, cos_long ) + LGAL;

  lon_g = tmp
  lon_g = lon_g % (2*np.pi)

  return lat_g, lon_g


# --- Sky localisation ----- #

def sigma_from_area90(area90):
    """Width [rad] of the symmetric 2D Gaussian whose 90% credible area is area90 [deg^2].
       For a 2D Gaussian the area enclosing a probability p is A_p = -2 pi sigma^2 ln(1-p),
       so A_90 = 2 pi ln(10) sigma^2."""
    return np.radians(np.sqrt(area90/(2*np.pi*np.log(10))))


def gaussian_scatter_sphere(lon, lat, sigma, rng, size=None):
    """Displaces the points (lon, lat) [rad] on the sphere by a symmetric 2D Gaussian of width sigma [rad]:
       the angular offset follows a Rayleigh distribution and the direction of the offset is uniform.
       Valid everywhere on the sphere, including near the poles."""

    rho = rng.rayleigh(scale=sigma, size=size) # angular distance from (lon, lat)
    beta = rng.uniform(0, 2*np.pi, size=size) # direction of the offset

    sin_lat = np.sin(lat)*np.cos(rho) + np.cos(lat)*np.sin(rho)*np.cos(beta)
    new_lat = np.arcsin(np.clip(sin_lat, -1, 1))
    new_lon = lon + np.arctan2(np.sin(beta)*np.sin(rho)*np.cos(lat), np.cos(rho) - np.sin(lat)*sin_lat)

    return new_lon % (2*np.pi), new_lat


def bbh_sky(gal_count_map_i, sel_map_i, Nbh_i, nside, rng):
    """Generate BBH positions (healpix pixels, galactic coordinates) following
    the galaxy counts in the i-th shell, weighted by the
    sky selection function of the detector network"""

    if np.any(np.isnan(gal_count_map_i)):
        print("Nan in galaxy count map")

    # Final BBH distribution in the sky
    weights = gal_count_map_i*sel_map_i
    final_pdf = weights/np.sum(weights) # pdf on the sky

    Bh_pix = rng.choice(np.arange(hp.nside2npix(nside)), size=Nbh_i, p=final_pdf)

    return Bh_pix


# --- Equations for lesing magnification ----- #

def sig_lens(z, OM, s8):
  res=(0.0004-0.00176*s8+(-0.035+s8*OM+0.0453*s8)*z)/((2.19+s8**2)*OM*z+3.19*np.exp(0.365/(0.193+z))) #checked
  return res

def mu3_lens(z, OM, s8):
  res=(s8**2*OM*z**2/(s8*np.sqrt(z)+1.1*z+(4.24*s8**2-OM**2)*OM*z**2+0.118*(1-s8)*z**3))**3 #checked
  return res

def mu4_lens(z, OM, s8):
  res=((-0.029+0.1*s8+0.47*OM*s8)*z/(np.exp((-0.029+0.1*s8+0.47*OM*s8)*z+0.021/(0.018+OM*s8*z))+0.3*z))**4 #checked
  return res

# from the magnitude moments above, we define de log-normal moments for the convergence:
def mu3_ln(z, OM, s8):
  a=np.log10(z)
  b=np.log10(s8)
  c=np.log10(OM)
  exp=a*(1.698-0.1979*c)-0.4081*b/(a-1.037)+0.2202/(a-1.037)- 0.4081*b**3*c-0.4081*b**2-0.8128*b*c+c-0.8128 #checked
  return 10**exp

def mu4_ln(z, OM, s8):
  a=np.log10(z)
  b=np.log10(s8)
  c=np.log10(OM)
  exp=-(b-0.2861)/(2.177*a+c-1.684)+1.721*a-0.4803*b**2-b*c+1.156*c-0.7733 #checked
  return 10**exp

def sig_ln(z, OM, s8):
  siglens=sig_lens(z, OM, s8)
  return (np.log(10)/5)*siglens #checked

def equations(p,z, OM, s8):
    frac1=mu3_lens(z, OM, s8)/mu3_ln(z, OM, s8)
    frac2=mu4_lens(z, OM, s8)/mu4_ln(z, OM, s8)
    x1 = p[0]
    return x1**4*(3+x1**2)-frac1-frac2+ x1**4*(3+16*x1**2+15*x1**4+6*x1**6+x1**8) #checked

def mu_ln(z, OM, s8):
  r_sol = fsolve(equations,np.array([1]),args=(z, OM, s8))[0]
  return np.abs(sig_ln(z, OM, s8)/r_sol)/4 #still not sure

def sig_gau2(z, OM, s8):
  res=np.log(1+sig_ln(z, OM, s8)**2/(mu_ln(z, OM, s8)**2)) #checked
  return res

def mu_gau(z, OM, s8):
  mu_ln2=mu_ln(z, OM, s8)**2
  res=np.log(mu_ln2/(1+sig_ln(z, OM, s8)**2/mu_ln2))/2 #cheked
  return res


# --- GW count maps from samples ----- #

def flatten_samples(samples_list):
    ra_all, dec_all, dl_all, obj_id_all = [], [], [], []

    for i, samples in enumerate(samples_list):
        n = len(samples['luminosity_distance'])
        if n == 0:
            continue

        ra_all.append(samples['lon'])
        dec_all.append(samples['lat'])
        dl_all.append(samples['luminosity_distance'])
        obj_id_all.append(np.full(n, i, dtype=int))

    return (
        np.concatenate(ra_all),
        np.concatenate(dec_all),
        np.concatenate(dl_all),
        np.concatenate(obj_id_all),
    )


def get_sky_map_all_sources(samples_list, nside, dl_bins):
    """Count map of GW sources in each luminosity distance bin, where each source
       is spread over the pixels and bins according to its samples."""

    Nbins = len(dl_bins) - 1

    npix = hp.nside2npix(nside)
    bbh_count_map = np.zeros((Nbins, npix), dtype=np.float32)

    # --- ORIGINAL sample counts per object
    N_per_obj_orig = np.array([
        len(s['luminosity_distance']) for s in samples_list
    ])

    if np.sum(N_per_obj_orig) == 0:
        return bbh_count_map

    # --- Flatten all samples
    ra, dec, dl, obj_id = flatten_samples(samples_list)

    # --- Remove samples outside the distance range
    mask = (dl >= dl_bins[0]) & (dl < dl_bins[-1])
    ra, dec, dl, obj_id = ra[mask], dec[mask], dl[mask], obj_id[mask]

    # --- Per-sample weights using ORIGINAL counts
    weights = 1.0 / N_per_obj_orig[obj_id]

    # --- Distance bin index
    bin_index = np.digitize(dl, dl_bins) - 1

    # --- Healpix pixel index
    pix_index = hp.ang2pix(
        nside,
        np.pi / 2.0 - dec,
        ra
    )

    # --- Single accumulation pass
    np.add.at(
        bbh_count_map,
        (bin_index, pix_index),
        weights
    )

    return bbh_count_map


class gw_forecast(object):
    ''' This class simulates GW sources observed by a detector network, given a forecast
    catalog of its detections. The sources are hosted by galaxies of the simulated galaxy
    catalogs, following the sky selection function of the network, and are observed with
    the distance and sky localisation errors of the forecast catalog. '''

    def __init__(self, params, rand=False):
        params = with_defaults(params) # default values of the optional parameters

        # Storage
        self.storage = params['storage']
        self.bdir = self.storage + params['bdir']
        self.gdir = self.storage + params['gdir']

        # Same folder naming as simulate_galaxies.sim_gal: {gdir}[_rand][_bias]
        if rand == True:
            self.gdir = f"{self.gdir}_rand"
            self.bdir = f"{self.bdir}_rand"

        self.kappa_dir = self.gdir # convergence does not depend on the host bias

        if params['bgw'] != 1:
            print_rank0("Using biased galaxy catalog for GW hosts")
            self.gdir = f"{self.gdir}_bias"

        # Forecast catalog
        self.catalog_file = self.storage + params['detector_forecast']
        if not os.path.isfile(self.catalog_file):
            self.catalog_file = params['detector_forecast']
        self.distance_unit = params['forecast_distance_unit']

        # Sky selection function of the detector network
        self.sky_selection = params['forecast_sky_selection']
        self.sel_dz = params['forecast_sel_dz']
        self.sel_nside = params['forecast_sel_nside']
        self.sel_smoothing = np.radians(params['forecast_sel_smoothing_deg'])

        # Errors
        self.err_nside = params['forecast_err_nside']
        self.lensing = params['include_lensing']
        self.save_samples = params['forecast_save_samples']

        # Simulation set up
        self.nside = params['nside']
        self.N_sim = params['N_sim']
        self.N_samples = params['N_samples']

        bin_edges = params['bin_edges']
        if type(bin_edges) != list:
            self.zb = np.load(bin_edges)
        else:
            self.zb = np.array(bin_edges)
        self.Nbins = len(self.zb)-1
        self.zbar = (self.zb[1:]+self.zb[:-1])/2 # intermediate redshifts of shells

        # Cosmology
        self.H0 = params['H0']
        self.OM = params['OM']
        self.omch2 = params['omch2']
        self.ombh2 = params['ombh2']
        self.As = params['As']
        self.ns = params['ns']
        self.tau = params['tau']

        self.rng = np.random.default_rng() # random generator

        if self.lensing == False:
            print_rank0("No lensing magnification")

        self._set_cosmology()
        self._load_catalog()
        self._bin_catalog()
        self._selection_maps()
        self._error_pixels()


    def _set_cosmology(self):
        """Background cosmology and sigma8 (for lensing magnification) from CAMB."""

        pars = camb.set_params(H0=self.H0, omch2=self.omch2, ombh2=self.ombh2, As=self.As,
                               ns=self.ns, tau=self.tau, NonLinear=camb.model.NonLinear_both)

        self.camb_back = camb.get_background(pars)

        pars.set_matter_power(redshifts=[0.0], kmax=2.0)
        results = camb.get_results(pars)
        self.s8 = float(results.get_sigma8()[0])

        # Bins in luminosity distance space
        self.dl_bins = self.camb_back.luminosity_distance(self.zb)


    def _load_catalog(self):
        """Read the forecast catalog and convert it to the quantities used in the simulation."""

        with open(self.catalog_file) as f:
            names = f.readline().lstrip('#').split()
        catalog = pd.read_csv(self.catalog_file, sep=r'\s+', skiprows=1, names=names, comment='#')

        missing = [c for c in ['ra', 'dec', 'err_Dl', 'err_area90'] if c not in catalog.columns]
        if 'z' not in catalog.columns and 'Dl' not in catalog.columns:
            missing.append('z (or Dl)')
        if missing:
            raise ValueError(f"Forecast catalog {self.catalog_file} is missing the columns: {missing}")

        if self.distance_unit not in ['Mpc', 'Gpc']:
            raise ValueError(f"forecast_distance_unit must be 'Mpc' or 'Gpc', not {self.distance_unit}")
        to_Mpc = 1000. if self.distance_unit == 'Gpc' else 1.

        if 'z' in catalog.columns:
            z = np.array(catalog['z'], dtype=float)
        else:
            # Redshifts from the luminosity distances in the fiducial cosmology
            z_grid = np.append(0., np.geomspace(1e-4, 100., 5000))
            z = np.interp(np.array(catalog['Dl'])*to_Mpc, self.camb_back.luminosity_distance(z_grid), z_grid)

        self.cat_z = z
        self.cat_ra = np.radians(np.array(catalog['ra'], dtype=float)) % (2*np.pi)
        self.cat_dec = np.radians(np.array(catalog['dec'], dtype=float))
        self.cat_err_dc = np.array(catalog['err_Dl'], dtype=float)*to_Mpc/(1+z) # error in comoving distance [Mpc]
        self.cat_sigma = sigma_from_area90(np.array(catalog['err_area90'], dtype=float)) # [rad]

        print_rank0(f"Loaded {len(z)} events from {self.catalog_file}")


    def _bin_catalog(self):
        """Number of GW events to simulate in each redshift bin."""

        bh_bin = np.digitize(self.cat_z, self.zb) - 1 # bin indexes of each BBH in the catalog, starting from 0
        bh_bin[(bh_bin < 0) | (bh_bin >= self.Nbins)] = -1 # outside the redshift range
        self.cat_bin = bh_bin

        self.Nbh_bincounts = np.bincount(bh_bin[bh_bin >= 0], minlength=self.Nbins) # N of BBHs in each bin

        print_rank0(f"Simulating {np.sum(self.Nbh_bincounts)} GW events inside the redshift bins "
              f"({np.sum(bh_bin < 0)} detections fall outside them)")


    def _selection_maps(self):
        """Sky selection function of the detector network in thick redshift bins, in galactic coordinates.
           It is estimated from the histogram of the positions of the events in the forecast catalog:
            - "ra_averaged": histogram averaged over right ascension, as the Earth rotation makes the
                             sensitivity depend only on declination for observations longer than a day;
            - "full": histogram in (ra, dec), e.g. for forecasts that include a non-uniform duty cycle;
            - "none": isotropic selection."""

        z0, zmax = self.zb[0], self.zb[-1]
        thick_edges = np.append(np.arange(z0, zmax, self.sel_dz), zmax)
        Nbins_thick = len(thick_edges)-1
        print_rank0('Number of thick bins:', Nbins_thick)

        npix = hp.nside2npix(self.nside)
        map_smoothed = np.ones((Nbins_thick, npix))

        if self.sky_selection not in ['ra_averaged', 'full', 'none']:
            raise ValueError(f"forecast_sky_selection must be 'ra_averaged', 'full' or 'none', not {self.sky_selection}")

        if self.sky_selection != 'none':

            # Equatorial coordinates of the centres of the pixels of the (galactic) output maps
            lon_g, lat_g = hp.pix2ang(self.nside, np.arange(npix), lonlat=True)
            ra_pix, dec_pix = galactic2equatorial(np.radians(lon_g), np.radians(lat_g))

            # Rings of iso-declination of the histogram maps
            npix_sel = hp.nside2npix(self.sel_nside)
            theta_sel, _ = hp.pix2ang(self.sel_nside, np.arange(npix_sel))
            _, ring = np.unique(np.round(theta_sel, 10), return_inverse=True)

            thick_bin = np.digitize(self.cat_z, thick_edges) - 1

            for j in range(Nbins_thick):

                in_bin = thick_bin == j
                if not np.any(in_bin):
                    warnings.warn(f'No forecast events in thick bin {j}: using an isotropic selection', stacklevel=2)
                    continue

                pix = hp.ang2pix(self.sel_nside, np.pi/2-self.cat_dec[in_bin], self.cat_ra[in_bin])
                hist_map = np.bincount(pix, minlength=npix_sel).astype(float) # equatorial coordinates

                if self.sky_selection == 'ra_averaged':
                    hist_map = (np.bincount(ring, weights=hist_map)/np.bincount(ring))[ring]

                map_gal = hp.get_interp_val(hist_map, np.pi/2-dec_pix, ra_pix)
                map_smoothed[j] = np.clip(hp.smoothing(map_gal, sigma=self.sel_smoothing, pol=False), 0, None)

        self.sel_maps = map_smoothed
        # thick bin of each thin bin
        self.thin2thick = np.clip(np.digitize(self.zbar, thick_edges) - 1, 0, Nbins_thick-1)

        if rank == 0:
            os.makedirs(self.bdir, exist_ok=True)
            np.savetxt(f'{self.bdir}/hist_maps_smooth.txt', map_smoothed)


    def _error_pixels(self):
        """For each redshift bin, the sky pixel (in equatorial coordinates) of every event of the
           forecast catalog. New events take the errors of a catalog event in the same bin and pixel,
           so that the errors keep their dependence on the sky position. The pixels are made larger
           (down to nside=1) until every pixel contains at least one catalog event."""

        self.err_pix = []

        for i in range(self.Nbins):

            idx = np.flatnonzero(self.cat_bin == i)
            nside_i = self.err_nside

            while True:
                pix = hp.ang2pix(nside_i, np.pi/2-self.cat_dec[idx], self.cat_ra[idx])
                if nside_i == 1 or np.unique(pix).size == hp.nside2npix(nside_i):
                    break
                nside_i = nside_i//2

            self.err_pix.append((idx, nside_i, pix))


    def _draw_errors(self, i, ra, dec):
        """Errors in comoving distance [Mpc] and sky position [rad] for new events at (ra, dec) [rad] in the bin i."""

        idx, nside_i, pix = self.err_pix[i]
        new_pix = hp.ang2pix(nside_i, np.pi/2-dec, ra)

        chosen = np.empty(len(new_pix), dtype=int)
        for p in np.unique(new_pix):
            here = new_pix == p
            pool = idx[pix == p]
            if pool.size == 0: # no catalog event in this pixel: take any event in the bin
                pool = idx
            chosen[here] = self.rng.choice(pool, size=np.sum(here))

        return self.cat_err_dc[chosen], self.cat_sigma[chosen]


    def _observe(self, i, Bh_pix, kappas):
        """ Adds lensing magnification and errors to the distances and positions of the
            GW sources in the bin i, and returns the list of their (Dl, lat, lon) samples,
            in galactic coordinates.

            Args:
                Bh_pix ('array'): healpix pixels where the GW sources are located.
                kappas ('array'): average convergence field for each pixel.
        """

        Nbh = len(Bh_pix)
        zbar_i = self.zbar[i]
        dz = self.zb[i+1]-self.zb[i]

        zs = zbar_i + self.rng.uniform(-0.5, 0.5, Nbh)*dz # uniform redshifts in each shell

        # Adding the magnification effect to the observed distances
        D = self.camb_back.comoving_radial_distance(zs, tol=0.0001)

        if self.lensing == True:
            # we draw a kappa for each object from a lognormal distribution
            s = np.sqrt(sig_gau2(zbar_i, self.OM, self.s8))
            scale = np.exp(mu_gau(zbar_i, self.OM, self.s8))
            loc = kappas[Bh_pix] - mu_ln(zbar_i, self.OM, self.s8)
            new_kappa = lognorm.rvs(s=s, loc=loc, scale=scale, size=Nbh, random_state=self.rng)

            mu = 1/(1-new_kappa)**2
            mag_D = D/np.sqrt(mu)
        else:
            mag_D = D

        # True positions (galactic) and errors, which depend on the equatorial position
        lon, lat = hp.pix2ang(self.nside, Bh_pix, lonlat=True)
        lon, lat = np.radians(lon), np.radians(lat)
        err_D, sigma = self._draw_errors(i, *galactic2equatorial(lon, lat))

        # Observed positions and distances
        new_lon, new_lat = gaussian_scatter_sphere(lon, lat, sigma, self.rng)
        new_D = self.rng.normal(mag_D, err_D)

        # Ensure new_D is positive
        neg = new_D < 0
        while np.any(neg):
            new_D[neg] = self.rng.normal(mag_D[neg], err_D[neg])
            neg = new_D < 0

        # Create 3D samples for each source
        samples_all = []

        for j in range(Nbh):

            dl_samples = (1+zs[j])*self.rng.normal(new_D[j], err_D[j], self.N_samples)
            dl_samples = dl_samples[dl_samples > 0]
            lon_samples, lat_samples = gaussian_scatter_sphere(new_lon[j], new_lat[j], sigma[j],
                                                               self.rng, size=len(dl_samples))

            samples = dict([])
            samples['luminosity_distance'] = dl_samples
            samples['lat'] = lat_samples
            samples['lon'] = lon_samples

            samples_all.append(samples)
            if len(dl_samples)==0:
                print('No samples generated for event', j+1)

            if self.save_samples == True:
                os.makedirs(f'{self.bdir}/samples', exist_ok=True)
                np.save(f'{self.bdir}/samples/samples{i+1}_{j+1}', samples)


        return samples_all


    def _load_kappa(self, k):
        """Convergence maps of the k-th galaxy simulation, one per redshift bin, read from
           {gdir}[_rand]/kappa_map{k}.npy: array of shape (Nbins, npix), same nside, ordering
           and coordinates as gal_count_map{k}.npy."""

        file = f'{self.kappa_dir}/kappa_map{k}.npy'
        npix = hp.nside2npix(self.nside)

        if not os.path.isfile(file):
            raise FileNotFoundError(f"include_lensing needs the convergence maps of galaxy simulation {k} "
                                    f"({file}), saved by simulate_galaxies.py. Set include_lensing: False "
                                    "if they are not available.")

        kappas = np.load(file)
        if kappas.ndim != 2 or kappas.shape[0] < self.Nbins or kappas.shape[1] != npix:
            raise ValueError(f"{file} has shape {kappas.shape}; expected ({self.Nbins}, {npix}) "
                             "(one convergence map per redshift bin)")

        return kappas


    def run(self):
        """Simulate the GW count maps for each galaxy simulation and store them."""

        npix = hp.nside2npix(self.nside)

        warn_existing_sims(self.bdir, self.N_sim, prefix="bbh_count_map", stacklevel=4)

        for k in tqdm(range(self.N_sim)[rank::size], disable=rank != 0):

            gal_count_map = np.load(f'{self.gdir}/gal_count_map{k+1}.npy')
            kappas = self._load_kappa(k+1) if self.lensing == True else [None]*self.Nbins

            hosts_map = np.zeros((self.Nbins, npix))
            bbh_count_map = np.zeros((self.Nbins, npix))

            for i in range(self.Nbins):

                # If there are no BBHs in this bin, skip
                if self.Nbh_bincounts[i]==0:
                    continue

                # Choose hosts from the galaxy map
                Bh_pix = bbh_sky(gal_count_map[i], self.sel_maps[self.thin2thick[i]],
                                 self.Nbh_bincounts[i], self.nside, self.rng)

                unique, counts = np.unique(Bh_pix, return_counts=True)
                hosts_map[i, unique] = counts

                # Add errors & get samples
                samples_list = self._observe(i, Bh_pix, kappas[i])

                bbh_count_map += get_sky_map_all_sources(samples_list, self.nside, self.dl_bins)

            os.makedirs(f'{self.bdir}', exist_ok=True)
            np.save(f'{self.bdir}/hosts_map{k+1}', hosts_map)

            np.save(f'{self.bdir}/bbh_count_map{k+1}', bbh_count_map)


def sim_bbh(params, rand=False):
    """Simulate BBH positions out of galaxy catalog, given a detector network forecast"""
    params = with_defaults(params) # default values of the optional parameters

    gw_forecast(params, rand=rand).run()
