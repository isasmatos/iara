# Generate galaxy catalog

import numpy as np
import camb
import glass.shells
import glass.fields
import glass.points
import glass.galaxies
import glass.lensing
import glass.observations
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import os, os.path
import pymaster as nmt
import healpy as hp
from mpi4py import MPI
from scipy.integrate import quad
# import glass.ext.camb
from cosmology.compat.camb import Cosmology
from scipy.stats import truncnorm
import transformcl


rank = MPI.COMM_WORLD.Get_rank() # index of the task
size = MPI.COMM_WORLD.Get_size() # total number of tasks
barrier = MPI.COMM_WORLD.Barrier

arcmin2_sky = 148510800
Mmin = -27
Mmax = -19
L0 = 3.0128*1e28 # Watts
L0 = 1 # luminosities will have units of L0 (above)

def bias_g(z, file=False):
    """Galaxy bias"""
    if file==False:
        return 0.66 + 1.62*z #0.971 + 1.042*z # Blue galaxies #(1+z)
    else:
        print("Reading bias from file")
        bias = np.load(file)
        return np.interp(z, bias[:,0], bias[:,1])
        

def schechter(L):
    """Unormalized Schechter function
       Gives the probability of a galaxy having intrinsic luminosity L"""
    
    Lmin = L0*10**(-2*Mmax/5)
    Lmax = L0*10**(-2*Mmin/5)
    alpha = -1.09
    Mstar = -23.39
    Lstar = L0*10**(-2*Mstar/5)
    
    return np.where((L > Lmin) & (L < Lmax), ((L/Lstar)**alpha)*np.exp(-L/Lstar), 0)

def cum_schechter(L, Lmin):
    """Cumulative distribution
    if probability density is the Schechter function"""
    
    return quad(schechter, Lmin, L)[0]
                         

def compute_gls(params, monopole=False, smoothing=False):

    """Compute Gaussian Cls. Set monopole=True for initial guess with large monopole in the first shell."""
    
    storage = params["storage"]
    matter_cls_path = storage+params["matter_cls"]

    lmax = params['lmax']
    nside = params['nside']
    N_samp = params['N_sim']
    corr = params['corr']
    bin_edges = params['bin_edges']

    if type(bin_edges) != list:
        bin_edges = np.load(bin_edges)
    else:
        bin_edges = np.array(bin_edges)

    # Arbitrary bin edges
    zb = bin_edges
    dz_arr = zb[1:]-zb[:-1]
    ws = glass.shells.tophat_windows(zb, dz=np.min(dz_arr)/100) # window function

    Nbins = len(zb)-1
    zbar = (zb[1:]+zb[:-1])/2

    # Load matter cls
    cls = np.load(matter_cls_path+'/cls.npy')
    # Compute Gaussian Cls for lognormal random fields for corr correlated shells, with discrete delta_m in pixels 

    # Regularize spectrum (if using very small bins such that correlations between adjacent bins are too large)
    #reg_gls = angst.regularized_spectra(gls, method='nearest')#, **method_kwargs)
    #gls = cls # Cls are of density contrast
    #gls = glass.fields.lognormal_gls(gls, nside=nside, lmax=lmax, ncorr=corr) # list of Nzbins*(Nzbins+1)/2 -> old version

    # set up lognormal matter fields for simulation
    fields = glass.lognormal_fields(ws)
    
    # Smooth out small scales and pad them to zero until 2*lmax
    if smoothing == True:
        ell = np.arange(lmax + 1)
        lpiv = max([lmax/3, 150]) # pivot scale after which attenuation starts
        warnings.warn("Attenuating Cls for l > %i"%lpiv, category=UserWarning, stacklevel=2)
        lpow = 5.0 # power of attenuation
        lpad = 2 * lmax # new lmax where cls reach zero
        fl2 = (1 - (ell / lmax) ** lpow) / (1 + (ell / lpiv) ** lpow)
        
        cls_mod = [np.pad(fl2 * cl, (0, lpad - cl.size)) for cl in cls]
    else:
        cls_mod = cls
    
    # Check angular correlation function is >-1
    for i in range(Nbins):
        cls_arr = glass.getcl(cls_mod,i,i,lmax=lmax)
        c_theta = transformcl.cltocorr(cls_arr)
        if np.any(c_theta < -1):
            mono = -(min(c_theta) + 1)*(4*np.pi)
            warnings.warn(r'$C(\theta) < -1$ at bin: %i'%i, category=UserWarning, stacklevel=2)
    
    if monopole==True:
        # Input large monopole to start (in shells where C(theta)<-1)
        cls_mod[0][0] = mono + 1.5
       
    Cls = glass.discretized_cls(cls_mod, nside=nside, lmax=2*lmax, ncorr=corr)
    
    # compute Gaussian spectra for lognormal fields from discretised spectra
    gls = glass.solve_gaussian_spectra(fields, Cls)
    print("Computed gls")
    
    if monopole==True:
        # now use this solution as initial guess for a zero-monopole solution
        gls[0], _, _ = glass.grf.solve(
            Cls[0],
            fields[0],
            fields[0],
            pad=2*cls[0].size,
            initial=gls[0],
            monopole=0.0,
        )
    
        print("Removed monopole and computed final Gls")
    
    cl_auto = []
    for i in range(Nbins):
        cl_auto.append(glass.getcl(gls, i, i, lmax=lmax))
    
    if np.any(np.array(cl_auto)<0):
        warnings.warn("Negative gls. Inspect cl_auto.", category=UserWarning, stacklevel=2)
    
    # Now Gls were confidently computed up to a lower lmax.
    lmax_gls = len(gls)-1
    
    if 3*nside-1 > lmax_gls:
        warnings.warn("nside may be too large for resolution of Gls (lmax_gls = %i). Increase lmax."%lmax_gls, category=UserWarning, stacklevel=2)

    os.makedirs(os.path.dirname(f'{matter_cls_path}/gls.npy'), exist_ok=True)
    arr = np.array(gls, dtype=object)
    np.save(f'{matter_cls_path}/gls', arr, allow_pickle=True)
    

def sim_gal(params, rand=False):
    """Generate galaxy catalogs. If rand = False (default), use matter power spectrum
    previously computed with camb. If True, use random distribution with Cl = 0.
    Use precomputed Gaussian Cls ("Gls") as input for GLASS mocks.
    If include_lensing == True, saves both the maps and the catalogs containing konvergence values.
    
    Output: files gal_count_map{k}.npy for k in [1, N_sim]
    Each of them are numpy arrays of shape (Nbins x Npixels) with galaxy counts"""
    
    storage = params["storage"]
    gdir = storage+params["gdir"]
    matter_cls_path = storage+params["matter_cls"]
    gal_dir = params['gal_dir']
    dndz_file = params["dndz"]

    nside = params['nside']
    N_samp = params['N_sim']
    corr = params['corr']
    bgw = params['bgw']
    bin_edges = params['bin_edges']

    if type(bin_edges) != list:
        bin_edges = np.load(bin_edges)
    else:
        bin_edges = np.array(bin_edges)
      
    # Arbitrary bin edges
    zb = bin_edges
    dz_arr = zb[1:]-zb[:-1]
    ws = glass.shells.tophat_windows(zb, dz=np.min(dz_arr)/100) # window function   
    Nbins = len(zb)-1

    # Fiducial cosmology
    H0 = params['H0']
    OM = params['OM']
    omch2 = params['omch2']
    ombh2 = params['ombh2']
    h = H0/100
    As = params['As']
    ns = params['ns']
    tau = params['tau']
    corr = params['corr']

    pars = camb.set_params(H0=H0, omch2=omch2, ombh2=ombh2,As=As,ns=ns,tau=tau,
                           NonLinear=camb.model.NonLinear_both)
    
    camb_back = camb.get_background(pars)
    cosmo = Cosmology(camb_back)

    # Load pre-computed Gaussian Cls
    gls = np.load(f'{matter_cls_path}/gls.npy', allow_pickle=True)
    if rand==True:
        gls = np.zeros(np.shape(gls))
        gdir = gdir+'_rand'

    # Objects to store galaxy catalogs
    catalog_G = np.zeros(len(zb)-1, dtype=[('LON', object), ('LAT', object),
                                           ('Z_TRUE', object), ('KAPPA', object)])#, ('SHEAR', object)])


    catalog_H = np.zeros(len(zb)-1, dtype=[('LON', object), ('LAT', object),
                                           ('Z_TRUE', object), ('KAPPA', object)])#, ('SHEAR', object)])


    for k in tqdm(range(N_samp)[rank::size]): # loop over simulations
        
        # Sample a lognormal random field from Gaussian cls (=gls)  
        matter = glass.fields.generate_lognormal(gls, nside, ncorr=corr) 
        # Get dndz
        if dndz_file==None:
            print("Using Smail distribution")

            try:
                z_mode, alpha, beta, Ngal = params["smail"]
            except:
                z_mode, alpha, beta, Ngal = 0.7, 0.8, 2.0, 6e7 # tuned for desi legacy [(1.0, 2.0, 1.5) -> used in Ferri et al. 2025]
            
            dndz_new = glass.observations.smail_nz(bin_edges, z_mode, alpha, beta, norm=Ngal/arcmin2_sky) # * (6*1e7/arcmin2_sky)
        
        else:
            print("Reading dndz from file")
            if params["multiple_surveys"] == True:
                dndz_new = {}
                for survey in params["footprint_dict"].keys():
                    print("File:", dndz_file[survey])
                    dndz_new[survey] = np.genfromtxt(storage+gal_dir+dndz_file[survey])
                    dndz_new[survey] = dndz_new[survey][:Nbins]
            else:
                dndz_new = np.genfromtxt(storage+gal_dir+dndz_file)
                dndz_new = dndz_new[:Nbins]

        # Set up convergence object
        if params["include_lensing"] == True:
            convergence = glass.lensing.MultiPlaneConvergence(cosmo)

        # For each shell generate galaxies from matter field
        for i, delta_i in enumerate(matter): 
        
            if i > Nbins-1:
                continue

            # Computing convergence
            if params["include_lensing"] == True:
                convergence.add_window(delta_i, ws[i])
                kappa_i = convergence.kappa
                catalog_G[i]['KAPPA'] = kappa_i
                #catalog_G[i]['SHEAR'] = gal_she
                print("Storing convergence.")


            if dndz_file==None: # smail
                zg_i, dndzg_i = glass.shells.restrict(zb, dndz_new, ws[i])  # smail
                ngal = np.trapezoid(dndzg_i, zg_i)

            else:
                if params["multiple_surveys"] == True:
                    # Number of galaxies in shell i per arcmin^2 for each survey
                    ngal = {}
                    for survey in params["footprint_dict"].keys():
                        ngal[survey] = dndz_new[survey][i]/arcmin2_sky

                else:
                    ngal = dndz_new[i]/arcmin2_sky
        

            if params["multiple_surveys"] == True:

                pos_gal_surveys = dict()

                for survey in params["footprint_dict"].keys():

                    print(f"Survey: {survey}")
                    # Read survey footprint
                    footprint = np.load(storage+gal_dir+params["footprint_dict"][survey])
                    # Read survey galaxy bias
                    bias_file = params["galaxy_bias"][survey]
                    bias = bias_g(zb, file=storage+gal_dir+bias_file)

                    # Checking amount of galaxies to simulate
                    if ngal[survey]*arcmin2_sky < 1 or np.isnan(dndz_new[survey][i]):
                        warnings.warn(f"No galaxies in bin {i}", category=UserWarning, stacklevel=2)
                        continue

                    # Apply footprint to delta_i
                    delta_i_survey = delta_i

                    # Generating the positions of galaxies from density field (ngal + contrast delta, with bias) in shell i
                    pos_gal_surveys[survey] = glass.points.positions_from_delta(ngal[survey], delta_i_survey,
                                                                                 bias=glass.points.effective_bias(zb, bias, ws[i]),
                                                                                  remove_monopole=True,
                                                                                    vis = footprint)

                pos_gal = (
                    item
                    for survey in params["footprint_dict"].keys()
                    for item in pos_gal_surveys[survey]
                )
                            
            else:
                # Read galaxy bias
                bias = bias_g(zb, file=False)
                # Checking amount of galaxies to simulate
                if ngal*arcmin2_sky < 1 or np.isnan(dndz_new[i]):
                    warnings.warn(f"No galaxies in bin {i}", category=UserWarning, stacklevel=2)
                    continue
                pos_gal = glass.points.positions_from_delta(ngal, delta_i, bias=glass.points.effective_bias(zb, bias, ws[i]), remove_monopole=True)

            # Store positions in catalog_G and catalog_H for this survey
            result_list = []
            for lon, lat, count in pos_gal:
                result_list.append((lon, lat, count))

            if result_list:
                gal_lons, gal_lats, gal_counts = zip(*result_list)
                gal_lons = np.concatenate(gal_lons)
                gal_lats = np.concatenate(gal_lats)
                gal_counts = np.sum(gal_counts)  # Not sure why to sum here
                gal_z = glass.galaxies.redshifts_from_nz(gal_counts, ws[i].za, ws[i].wa)
            else:
                gal_lons, gal_lats, gal_z = [], [], []

            catalog_G[i]['LON'] = gal_lons  # deg
            catalog_G[i]['LAT'] = gal_lats  # deg
            catalog_G[i]['Z_TRUE'] = gal_z

            # Hosts map --------------
            if bgw != 1: # Generate a second catalog with different galaxy bias
            
                pos_hosts = glass.points.positions_from_delta(ngal, delta_i,bias=glass.points.effective_bias(zb, bgw*bias_g(zb), ws[i]))#, remove_monopole=True)
                result_list_hosts = []

                for lon, lat, count in pos_hosts:
                    # lon and lat are arrays with the positions of each galaxy that was sampled; 
                    # count is the total number of galaxies sampled in the shell
                    result_list_hosts.append((lon, lat, count))
            
                if result_list_hosts:
                    gal_lons, gal_lats, gal_counts = zip(*result_list_hosts)
                    gal_lons = np.concatenate(gal_lons)
                    gal_lats = np.concatenate(gal_lats)
            
                    gal_counts = np.sum(gal_counts) # --->  Not sure why to sum here
                    gal_z = glass.galaxies.redshifts_from_nz(gal_counts, ws[i].za, ws[i].wa)

                else:
                    gal_lons, gal_lats, gal_z = [], [], []
            
                catalog_H[i]['LON'] = gal_lons # deg
                catalog_H[i]['LAT'] = gal_lats # deg
                catalog_H[i]['Z_TRUE'] = gal_z
    
        gal_count_map = np.zeros(shape=(Nbins, hp.nside2npix(nside)))
        hosts_count_map = np.zeros(shape=(Nbins, hp.nside2npix(nside)))

        # Store maps in gdir
        for i in range(Nbins):

            gal_pix_index  = hp.ang2pix(nside, catalog_G[i]['LON'], catalog_G[i]['LAT'], lonlat=True)
            unique, counts = np.unique(gal_pix_index, return_counts=True)
            gal_count_map[i,unique] = counts # galaxy counts N in each redshift bin i    
        
        os.makedirs(os.path.dirname(f'{gdir}/gal_count_map{k+1}.npy'), exist_ok=True)
        
        if f'gal_count_map{k+1}.npy' in os.listdir(gdir):
            warnings.warn(f'Simulation {k+1} already there. Overwritting.', category=UserWarning, stacklevel=2)

        np.save(f'{gdir}/gal_count_map{k+1}.npy', gal_count_map)

        # For lensing we need to save the catalog and kappa values
        if params["include_lensing"] == True:
            os.makedirs(os.path.dirname(f'{gdir}/catalog_G{k+1}.npy'), exist_ok=True)
            np.save(f'{gdir}/catalog_G{k+1}', catalog_G)

        if bgw != 1: # Store maps of hosts in gdir_bias
        
            for i in range(Nbins):

                gal_pix_index  = hp.ang2pix(nside, catalog_H[i]['LON'], catalog_H[i]['LAT'], lonlat=True)
                unique, counts = np.unique(gal_pix_index, return_counts=True)
                hosts_count_map[i,unique] = counts # galaxy counts N in each redshift bin i    
         
            os.makedirs(os.path.dirname(f'{gdir}_bias/gal_count_map{k+1}.npy'), exist_ok=True)
            np.save(f'{gdir}_bias/gal_count_map{k+1}.npy', hosts_count_map) 


def selection_Poisson(params, rand=False):
    """Takes simulated complete galaxy map and obtain its incomplete version
    by multiplying the counts by the completeness map.
    It also masks the catalog.
    Then draw final galaxy counts from Poissonian distribution centered on it.
    The last step is to add photometric redshift uncertainties
    for a fraction of galaxies.
    """
    
    storage = params["storage"]
    gdir = storage+params["gdir"]
    gal_dir = storage+params["gal_dir"]
    completeness_path = gal_dir+params["completeness"]
    mask_path = gal_dir+params["mask"]
    Nphotoz_path = gal_dir+params["Nphotoz"]
    ng_obs_path = gal_dir+params["gal_map"]

    Nsim = params['N_sim']
    photoz = params['photoz']
    nside = params['nside']
    bin_edges = params['bin_edges']

    if rand==True:
        gdir = gdir+'_rand'

    if type(bin_edges) != list:
        bin_edges = np.load(bin_edges)
    else:
        bin_edges = np.array(bin_edges)
      
    # Arbitrary bin edges
    zb = bin_edges
    zmax = zb[-1]
    Nbins = len(zb)-1
    
    completeness = np.load(completeness_path)
    gmask = np.load(mask_path)
    Nphotoz = np.load(Nphotoz_path)
    Nphotoz = np.trunc(Nphotoz).astype(int)[:Nbins]

    ng_obs = np.load(ng_obs_path)
    ng_obs = np.sum(ng_obs,axis=1)[:Nbins]
    Nspec = np.trunc(ng_obs - Nphotoz).astype(int)
 
    selection = completeness*gmask

    for k in range(Nsim)[rank::size]:
            
            # Completeness + mask
            g_map = np.load(f'{gdir}/gal_count_map{k+1}.npy', allow_pickle=True)
            g_map_incomp = g_map*selection[:np.shape(g_map)[0]]
            # Poisson
            g_map_new = np.random.poisson(g_map_incomp)

            # Photo z
            gal_map_photoz = np.copy(g_map_new)
            npix = hp.nside2npix(nside)
            g_map_photoz_final = g_map_new.astype(float)
            g_map_final = g_map_new.astype(float)
            #print('N_gal before photoz = ',np.sum(g_map_new, axis=1))

            for i in range(Nbins):

                if np.sum(g_map_new[i]) == 0:
                    continue

                # Split galaxy map into spec and photo galaxies. What is sure is Nspec(z)
                arr = np.repeat(np.arange(npix), g_map_new[i]) # galaxy pixels
                gal_ids = np.arange(len(arr))  # unique galaxy IDs
                
                if len(arr) == 0:
                    print(f"Bin {i}: no galaxies available at all.", flush=True)
                        
                if Nspec[i] > len(arr):
                    print(f"Not enough galaxies with spectrum in bin {i} in sim {k}: "
                          f"requested {Nspec[i]}, available {len(arr)}", flush=True)
                    Nspec[i] = len(arr)
                    
                gal_spec = np.random.choice(gal_ids, size=Nspec[i], replace=False)
                p_spec = arr[gal_spec]

                pixels_spec, counts_spec = np.unique(p_spec, return_counts=True)                 
                gal_map_photoz[i, pixels_spec] = g_map_new[i, pixels_spec] - counts_spec # galaxies with photo-z only
                
                Nphotoz_new = np.trunc(np.sum(gal_map_photoz[i])).astype(int)
                                        
                # Sample true z from truncated Gaussian
                #zt = np.ones(Nphotoz_new)*(zb[i]+zb[i+1])/2
                zt = np.random.uniform(zb[i],zb[i+1], Nphotoz_new)
                
                min = (0 - zt) / photoz
                max = (zmax + 10*photoz - zt) / photoz
                z_new = truncnorm.rvs(min,max,loc=zt, scale=photoz, size=Nphotoz_new)
                z_new = z_new[z_new<zmax] # galaxies that are moved to beyond zmax
                Nphotoz_new2 = len(z_new)

                bins = np.digitize(z_new, zb)-1
                b, c = np.unique(bins, return_counts=True) # new counts
 
                # Total number of new galaxies per z-bin
                photoz_dist = np.zeros(Nbins, dtype=int)
                photoz_dist[b] = c

                # Stack photo-z of galaxies in the same pixel
                counts = gal_map_photoz[i].astype(int)
                
                # Rebuild pixel indices for the galaxies with photo-z (these are the "photo" ones)
                pix_ids = np.repeat(np.arange(npix), counts)
                
                if Nphotoz_new2 < Nphotoz_new:
                    keep_indices = np.random.choice(Nphotoz_new, size=Nphotoz_new2, replace=False)
                    pix_ids = pix_ids[keep_indices]
            
                # Now fill the (bin, pixel) histogram
                photoz_stack = np.zeros((Nbins, npix), dtype=int)                
                # Assign each galaxy to its (bin, pixel)
                np.add.at(photoz_stack, (bins, pix_ids), 1)

                # Build new observed galaxy map
                g_map_photoz_final[i,:] -= counts
                g_map_photoz_final += photoz_stack

            ng_obs_sim = np.sum(g_map_photoz_final, axis=1)
            if np.any(ng_obs_sim < Nphotoz):
                print(f'Not enough galaxies in bin {np.where(ng_obs_sim < ng_obs)}')

            bins_exceed = np.where(ng_obs_sim > ng_obs)[0]
            #print(bins_exceed)
            for bin in bins_exceed:
                #print(bin)
                n_exceed = int(ng_obs_sim[bin]-Nphotoz[bin])#ng_obs[bin])
                #print(np.shape(n_exceed))
                #print("Too many galaxies =", n_exceed)
                arr2 = np.repeat(np.arange(npix), g_map_photoz_final[bin].astype(int))
                gal_ph_ids = np.arange(len(arr2))
                gal_out = np.random.choice(gal_ph_ids, size=n_exceed, replace=False)
                p_out = arr2[gal_out]
                pix, count = np.unique(p_out, return_counts=True)
                g_map_photoz_final[bin,pix] -= count

            g_map_final -= gal_map_photoz 
            g_map_final += g_map_photoz_final

          
            os.makedirs(os.path.dirname(f'{gdir}_obs/gal_count_map{k+1}.npy'), exist_ok=True)
            np.save(f'{gdir}_obs/gal_count_map{k+1}.npy', g_map_final)




def selection_Poisson_dndz(params, rand=False):
    """Takes simulated complete galaxy map and produces an observed version
    applying completeness, masking, Poisson sampling, and photo-z scatter
    via a dndz transfer matrix built from a continuous dndz(z_true | z_obs) curve.

    Reads params["dndz_photoz"]: a text file whose first column is z_arr and
    whose remaining Nbins columns are dndz(z_true | z_obs=bin_j) for each
    observed bin j.  The (Nbins, Nbins) scatter matrix is built by integrating
    each column over the true-z bin intervals (CDF differences), then
    column-normalising to obtain p(z_obs=j | z_true=i).
    """

    storage = params["storage"]
    gdir = storage + params["gdir"]
    gal_dir = storage + params["gal_dir"]
    completeness_path = params["completeness"]
    mask_path = params["mask"]
    dndz_photoz_dict = params["dndz_photoz"]
    fp_dict_path = params["footprint_dict"]

    # Load footprint dictionary for each survey
    fp_dict = dict([])
    for survey in fp_dict_path.keys():
        fp_dict[survey] = np.load(gal_dir + fp_dict_path[survey])

    Nsim = params['N_sim']
    nside = params['nside']
    bin_edges = params['bin_edges']

    if rand:
        gdir = gdir + '_rand'

    if type(bin_edges) != list:
        bin_edges = np.load(bin_edges)
    else:
        bin_edges = np.array(bin_edges)

    zb = bin_edges
    Nbins = len(zb) - 1
    npix = hp.nside2npix(nside)

    completeness = np.load(gal_dir + completeness_path)
    if mask_path is not None:
        print("Loading mask")
        gmask = np.load(gal_dir + mask_path)
    else:
        print("Using superposition of footprints as mask")
        gmask = np.sum(list(fp_dict.values()), axis=0) # union of all survey footprints
        gmask = np.where(gmask > 0, 1, 0)  # ensure binary mask

    if gmask.ndim == 1:
        # If mask is a single one for all redshifts
        selection = completeness * gmask[None, :]
    elif gmask.ndim == 2:
        # if mask is redshift dependent
        selection = completeness * gmask
    else:
        warnings.warn(f'Shapes of completeness and mask arrays are incompatible.',
                                  category=UserWarning, stacklevel=2)


    print("Built selection function from completeness and mask.")

    # Build one scatter matrix per region
    scatter_dict = {survey: np.load(gal_dir + dndz_photoz_dict[survey])
                for survey in fp_dict.keys()}

    for k in range(Nsim)[rank::size]:

        # Load complete map
        g_map = np.load(f'{gdir}/gal_count_map{k+1}.npy', allow_pickle=True)

		# Apply selection function
        g_map_incomp = g_map * selection[:np.shape(g_map)[0]]
        g_map_new = np.random.poisson(g_map_incomp)

        g_map_obs = np.zeros((Nbins, npix), dtype=float)

        for i in range(Nbins):
            if np.sum(g_map_new[i]) == 0:
                continue

            # Pixel index for every galaxy in true bin i
            pix_list = np.repeat(np.arange(npix), g_map_new[i].astype(int))

            for region, pix_mask in fp_dict.items():

                sel = np.where(pix_mask[pix_list]>0, True, False)

                if not np.any(sel):
                    continue

                obs_bins = np.random.choice(
                    Nbins,
                    size=sel.sum(),
                    p=scatter_dict[region][:, i]
                )

                np.add.at(g_map_obs, (obs_bins, pix_list[sel]), 1)

        os.makedirs(os.path.dirname(f'{gdir}_obs/gal_count_map{k+1}.npy'), exist_ok=True)

        if f'gal_count_map{k+1}.npy' in os.listdir(f'{gdir}_obs'):
            warnings.warn(f'Simulation {k+1} already there. Overwriting.',
                          category=UserWarning, stacklevel=2)

        np.save(f'{gdir}_obs/gal_count_map{k+1}.npy', g_map_obs)



def gal_cls(params, gdir):
    
    """Computes pseudo-cls given galaxy catalogs.
    For galaxies, masks 1/3 of the sky. Uses bandpowers of width l = bs."""
    
    storage = params["storage"]
    gal_dir = storage+params["gal_dir"]
    mask_path = gal_dir+params["mask"]

    #zmax = params['zmax']
    #dz = eval(params['dz_thin'])#params['dz']
    nside = params['nside']
    lmax = 3*nside-1 #params['lmax']
    N_samp = params['N_sim']
    bs = params['bs']

    # Binning
    
    bin_edges = params['bin_edges']

    if type(bin_edges) != list:
        bin_edges = np.load(bin_edges)
    else:
        bin_edges = np.array(bin_edges)

    zb = bin_edges
    Nbins = len(zb)-1

    # Pre-computing the coupling matrices

    lmax_cls = lmax
    b = nmt.NmtBin.from_lmax_linear(lmax_cls, bs) # binning in bandpowers, starts from ell=2 and removes last bandpowers if bin is incomplete
    n_bands = b.get_n_bands()

    # Get redshift dependent full galaxy mask
    gmask = np.load(mask_path)

    # Computing coupling matrices
    w_gg = np.empty(Nbins, dtype='object')
    
    for i in range(Nbins):
        gmask_i = nmt.NmtField(gmask[i,:], None, spin=0, lmax=lmax, masked_on_input=False)
        w_gg[i] = nmt.NmtWorkspace() # objects to store mode-coupling matrices
        w_gg[i].compute_coupling_matrix(gmask_i, gmask_i, b)

    #print("Computed coupling matrices")
    # Computing pseudo-cls
    cl = []
    print(N_samp)

    for k in tqdm(range(N_samp)):#[rank::size]):

        Pcls_gg = np.zeros(shape=(n_bands, Nbins, Nbins), dtype='float32')
        gal_delta_map = np.zeros(shape=(Nbins, hp.nside2npix(nside)))
        alm_g = np.zeros(shape=(Nbins, np.sum(np.arange(lmax_cls+1)+1)), dtype='complex')

        # Get galaxy count map from simulations
        gal_count_map = np.load(f'{gdir}/gal_count_map{k+1}.npy')
        gal_count_map = gal_count_map[:Nbins]
        
        for i in range(Nbins):
            map_masked = gal_count_map[i] * gmask[i]
            alm_g[i] = hp.map2alm(map_masked,lmax=lmax_cls) # 36s for both alms
      
        #print('computing cls')
        cls_gg = hp.alm2cl(alm_g, lmax=lmax_cls)

        # Inverting the pseudo-cls
        #print('inverting pseudo-cls')
            
        idx = 0
        for n in range(Nbins):
            for i in range(Nbins - n):
                j = n + i
                Pcls_gg[:,i,j] = Pcls_gg[:,j,i] = w_gg[i].decouple_cell(cls_gg[idx][np.newaxis,:], cl_bias=None)
                idx += 1

        cl.append(Pcls_gg)
            
        #print('saving cls')
        #os.makedirs(os.path.dirname(f'{cls_dir}/Cls_gg{k+1}.npy'), exist_ok=True)
        #np.save(f'{cls_dir}/Cls_gg{k+1}', Pcls_gg)
        
    return np.array(cl)