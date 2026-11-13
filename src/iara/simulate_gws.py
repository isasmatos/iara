"""GW simulations from real LVK events.

:func:`sim_gws` places each LVK event in a host drawn from the galaxy maps and moves its posterior
samples (distance and sky position) accordingly, giving the GW count maps of each simulation.
:func:`insert_bright_siren` adds a bright siren. For forecasts, see :mod:`iara.simulate_gws_forecasts`.
"""

import numpy as np
import healpy as hp
import camb
import glass.shells
# import glass.ext.camb
from tqdm import tqdm
import warnings
import os, os.path
# from pathlib import Path
import arviz as az
import h5py
import yaml
# from os.path import isfile, join
from mpi4py import MPI
from iara.simulate_galaxies import warn_existing_sims
from iara.defaults import with_defaults

warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

rank = MPI.COMM_WORLD.Get_rank() # index of the task
size = MPI.COMM_WORLD.Get_size() # total number of tasks
barrier = MPI.COMM_WORLD.Barrier


def print_rank0(*args, **kwargs):
    """Print once (from rank 0), not once per MPI process."""
    if rank == 0:
        print(*args, **kwargs)


def sample_hosts(gal_count_map, Nbh_i, nside): # IM: modified to take directly the galaxy count map
    """Generate BBH positions in galactic coordinates following
    the galaxy sky distribution in the i-th shell"""

    npix = hp.nside2npix(nside)
    arr = np.repeat(np.arange(npix), gal_count_map.astype(int))
    Bh_pix = np.random.choice(arr, size=Nbh_i, replace=False)

    return Bh_pix

def max_posterior(dl_samples, ra_samples, dec_samples, dl_bins, nside):
    """Get position (bin, pixel) of the maximum of the posterior samples."""
    Nbins = len(dl_bins) - 1
    bin_index = np.digitize(dl_samples, dl_bins) - 1
    best_count, best_bin, best_pix = 0, 0, 0
    for i in range(Nbins):
        mask = (bin_index == i)
        if not mask.any():
            continue
        pix = hp.ang2pix(nside, np.pi/2 - dec_samples[mask], ra_samples[mask])
        vals, counts = np.unique(pix, return_counts=True)
        peak = counts.argmax()
        if counts[peak] > best_count:
            best_count = counts[peak]
            best_bin = i
            best_pix = vals[peak]
    return best_bin, best_pix

def resamp(dl_samples, ra_samples, dec_samples, ddec):
    """Build new samples from previous ones correcting
       the prior in declination"""

    N = len(dl_samples)

    old_prior = lambda dec : np.cos((dec-ddec + np.pi/2) % np.pi - np.pi/2)
    new_prior = lambda dec : np.cos(dec)

    weights = new_prior(dec_samples)/old_prior(dec_samples)
    weights /= np.sum(weights)
    idx = np.random.choice(N, size=N, p=weights)

    return dl_samples[idx], ra_samples[idx], dec_samples[idx]

def _load_event_samples(lvk_maps_path, lvk_samples_path):
    """Pre-load posterior samples for all GW events that have a skymap.
    Keyed by map event name so bh_pix_dict lookups are direct.
    Called once before the simulation loop."""
    list_ev = [os.path.splitext(f)[0] for f in sorted(os.listdir(lvk_maps_path)) if 'GW' in f]
    all_samples = {}
    for fname in sorted(os.listdir(lvk_samples_path)):
        h5_key = os.path.splitext(fname)[0]
        # Match to a map event name (substring either way, same logic as original)
        map_name = next((ev for ev in list_ev if h5_key in ev or ev in h5_key), None)
        if map_name is None:
            continue
        if "170817" in map_name or "191219" in map_name:
            continue
        with h5py.File(os.path.join(lvk_samples_path, fname), mode="r") as f:
            if "190425" in h5_key:
                samples = f['C01:IMRPhenomPv2_NRTidal:LowSpin']['posterior_samples']
            elif "200115_042309" in h5_key:
                samples = f['C01:IMRPhenomXPHM:LowSpin']['posterior_samples']
            elif "200105_162426" in h5_key:
                samples = f['C01:IMRPhenomXPHM']['posterior_samples']
            else:
                try:
                    samples = f['C00:IMRPhenomXPHM-SpinTaylor']['posterior_samples']
                    print(f'Using waveform C00:IMRPhenomXPHM-SpinTaylor for {map_name}')
                except KeyError:
                    samples = f['C01:IMRPhenomXPHM']['posterior_samples']
                    print(f'Using waveform C01:IMRPhenomXPHM for {map_name}')
            all_samples[map_name] = {
                'dl': samples['luminosity_distance'][()],
                'ra': samples['ra'][()],
                'dec': samples['dec'][()],
            }
    return all_samples


def lvk_move(storage, all_samples, nside, bh_pix_dict, dl_bins, bdir, sim):
    """Move LVK event posterior samples to host galaxy positions and build count map.

    all_samples : dict {map_event_name: {'dl', 'ra', 'dec'}} from _load_event_samples.
    bh_pix_dict : dict {map_event_name: host_pixel}, built per simulation in sim_gws.
    """

    Nbins = len(dl_bins) - 1
    bbh_map = np.zeros(shape=(Nbins, hp.nside2npix(nside)))

    for event_key, samp in all_samples.items():

        if event_key not in bh_pix_dict:
            continue

        bbh_map_indv = np.zeros(shape=(Nbins, hp.nside2npix(nside)))

        dl_samples = samp['dl'].copy()
        ra_samples = samp['ra'].copy()
        dec_samples = samp['dec'].copy()
        N = len(dl_samples)

        # Get maximum of posterior
        _, pix_max = max_posterior(dl_samples, ra_samples, dec_samples, dl_bins, nside)
        dec_mean, ra_mean = hp.pix2ang(nside, pix_max)
        dec_mean = np.pi/2 - dec_mean

        # Move samples to host galaxy position
        theta_host, ra_host = hp.pix2ang(nside, bh_pix_dict[event_key])
        dec_host = np.pi/2 - theta_host
        dra = ra_host - ra_mean
        ddec = dec_host - dec_mean
        ra_samples_new = (ra_samples + dra) % (2*np.pi)
        dec_samples_new = (dec_samples + ddec + np.pi/2) % np.pi - np.pi/2
        dl_samples_new = dl_samples

        # Reweighting correcting for prior in declination
        dl_samples_new, ra_samples_new, dec_samples_new = resamp(
            dl_samples_new, ra_samples_new, dec_samples_new, ddec)

        # Now move samples away from galaxy position by 1sigma
        # Get error bars
        hdi_prob = 0.68

        hdi = az.hdi(ra_samples_new - np.pi, hdi_prob=hdi_prob, circular=True) # circular between -pi and pi
        hdi_ra = hdi + np.pi
        if hdi_ra[-1] < hdi_ra[0]:
            hdi_ra[-1] = hdi_ra[-1] + 2*np.pi
        sigma_ra = (hdi_ra[-1] - hdi_ra[0]) / 2

        hdi = az.hdi(dec_samples_new*2, hdi_prob=hdi_prob) # easier to first deal with 2*dec
        if hdi[-1] < hdi[0]:
            hdi[-1] = hdi[-1] + 2*np.pi
        sigma_dec = (hdi[-1] - hdi[0]) / 2
        sigma_dec = sigma_dec / 2

        # Get new central values in ra, dec
        ra_central = np.random.normal(ra_host, sigma_ra) % (2*np.pi)
        dec_central = (np.random.normal(dec_host, sigma_dec) + np.pi/2) % np.pi - np.pi/2

        # Get maximum of new posterior samples
        _, pix_max = max_posterior(dl_samples_new, ra_samples_new, dec_samples_new, dl_bins, nside)
        dec_mean_new, ra_mean_new = hp.pix2ang(nside, pix_max)
        dec_mean_new = np.pi/2 - dec_mean_new

        # Move to new final positions in ra, dec
        dl_samples_final = dl_samples_new
        dra = ra_central - ra_mean_new
        ra_samples_final = (ra_samples_new + dra) % (2*np.pi)
        ddec = dec_central - dec_mean_new
        dec_samples_final = (dec_samples_new + ddec + np.pi/2) % np.pi - np.pi/2

        # Reweighting with modified prior in declination
        dl_samples_final, ra_samples_final, dec_samples_final = resamp(
            dl_samples_final, ra_samples_final, dec_samples_final, ddec)

        # Cut radial bins beyond zmax
        in_range = dl_samples_final < dl_bins[-1]
        dec_samples_final = dec_samples_final[in_range]
        ra_samples_final = ra_samples_final[in_range]
        dl_samples_final = dl_samples_final[in_range]

        N = len(dl_samples_new) # number of samples before tail cutting

        # Split new samples into distance bins and build count map
        bin_index = np.digitize(dl_samples_final, dl_bins) - 1
        try:
            for j in range(Nbins):
                in_bin = (bin_index == j)
                if not in_bin.any():
                    continue
                bbh_pix_index = hp.ang2pix(nside, np.pi/2 - dec_samples_final[in_bin],
                                            ra_samples_final[in_bin])
                unique, counts = np.unique(bbh_pix_index, return_counts=True)
                bbh_map[j, unique] += counts / N
                bbh_map_indv[j, unique] += counts / N
        except TypeError:
            print(f'Error in event {event_key}')

        os.makedirs(os.path.dirname(f'{bdir}/sim{sim+1}/{event_key}.npy'), exist_ok=True)
        np.save(f'{bdir}/sim{sim+1}/{event_key}.npy', bbh_map_indv)

    return bbh_map


def sim_gws(params, rand=False):
    """Simulate BBH positions out of galaxy catalog
        """
    params = with_defaults(params) # default values of the optional parameters
    
    storage = params["storage"]
    lvk_maps_path = storage+params["lvk_maps"]
    lvk_samples_path = storage+params["lvk_samples"]
    bdir = storage+params['bdir']
    gdir = storage+params['gdir']
    bgw = params['bgw']

    if rand==True:
        gdir = gdir+'_rand'
        bdir = bdir+'_rand'
        
    if bgw != 1:
        print_rank0("Using biased galaxy catalog for GW hosts")
        gdir = f"{gdir}_bias"

    # Bins
    nside = params['nside']
    N_samp = params['N_sim']
    bin_edges = params['bin_edges']

    if type(bin_edges) != list:
        bin_edges = np.load(bin_edges)
    else:
        bin_edges = np.array(bin_edges)

    zb = bin_edges

    # Fiducial cosmology
    H0 = params['H0']
    omch2 = params['omch2']
    ombh2 = params['ombh2']
    h = H0/100
    As = params['As']
    ns = params['ns']
    tau = params['tau']

    pars = camb.set_params(H0=100*h, omch2=omch2, ombh2=ombh2, As=As,ns=ns,tau=tau,
                           NonLinear=camb.model.NonLinear_both)

    camb_back = camb.get_background(pars)
    
    # Bins in distance space
    dl_bins = camb_back.luminosity_distance(zb)
    Nbins = len(dl_bins)-1

    if params['multiple_surveys'] == True:
        # Loading list of surveys for each event
        with open(storage+params['gal_dir']+params['survey_dict'], 'r') as stream:
            survey_dict = yaml.safe_load(stream)
        # Loading survey footprints
        footprint_dict = params['footprint_dict'].copy()
        for survey in footprint_dict.keys():
            footprint_dict[survey] = np.load(storage+params['gal_dir']+footprint_dict[survey])

    # Pre-load all H5 posterior samples once — shared across all simulations
    print_rank0("Pre-loading event posterior samples...")
    all_samples = _load_event_samples(lvk_maps_path, lvk_samples_path)
    print_rank0(f"Loaded {len(all_samples)} events.")

    warn_existing_sims(bdir, N_samp, prefix="bbh_count_map")

    for k in tqdm(range(N_samp)[rank::size], disable=rank != 0):

        gal_count_map = np.load(f'{gdir}/gal_count_map{k+1}.npy')

        # Pick a distance bin for each event from its distance posterior
        gw_maps = {}
        map_event_names = []
        bin_index = []
        footprint_index = [] # when having multiple surveys

        for file in sorted(os.listdir(lvk_maps_path)): # alphabetical order
            #if '190814' not in file: # For generating GW190814 only - COMMENT THIS OTHERWISE
            #    continue
            if 'GW' in file:
                gw_maps[file] = np.load(f'{lvk_maps_path}/{file}')
                map_event_names.append(os.path.splitext(file)[0])
                p = np.sum(gw_maps[file], axis=1) / np.sum(gw_maps[file])
                bin_index.append(np.random.choice(range(Nbins), p=p))
                if params['multiple_surveys'] == True:
                    surveys = list(survey_dict[file[:-4]].keys())
                    overlaps = list(survey_dict[file[:-4]].values())
                    chosen_survey = str(np.random.choice(surveys, p=overlaps/np.sum(overlaps)))
                    footprint_index.append(footprint_dict[chosen_survey])

        bin_index = np.array(bin_index)

        # Sample host galaxy pixel for each event from its assigned redshift bin,
        # then collect into bh_pix_dict for a single lvk_move call
        bh_pix_dict = {}
        for i in range(Nbins):
            events = np.where(bin_index == i)[0]
            if len(events) == 0:
                warnings.warn(f"No events in bin {i}", category=UserWarning, stacklevel=2)
                continue
            if params['multiple_surveys'] == True:
                Bh_pix = np.array([
                    sample_hosts(gal_count_map[i] * footprint_index[ev].astype(float), 1, nside)[0]
                    for ev in events
                ])
            else:
                Bh_pix = sample_hosts(gal_count_map[i], len(events), nside)
            for j, ev in enumerate(events):
                bh_pix_dict[map_event_names[ev]] = Bh_pix[j]

        # Process all events in a single call — H5 data already in memory
        bbh_count_map = lvk_move(storage, all_samples, nside, bh_pix_dict, dl_bins, bdir, k)

        os.makedirs(os.path.dirname(f'{bdir}/bbh_count_map{k+1}.npy'), exist_ok=True)
        np.save(f'{bdir}/bbh_count_map{k+1}', bbh_count_map)            


def insert_bright_siren(params):
    """Take simulated galaxy maps and simulate the position of the bright siren
       on top of it"""
    params = with_defaults(params) # default values of the optional parameters
    
    storage = params["storage"]
    gal_dir = params['gal_dir']
    mask_path = gal_dir+params['mask']
    lvk_maps_path = storage+params["lvk_maps"]
    bdir = storage+params['bdir']
    gdir = storage+params['gdir']
    bgw = params['bgw']

    if bgw != 1:
        print_rank0("Using biased galaxy catalog for GW hosts")
        gdir = f"{gdir}_bias"
        
    zmax = params['zmax']
    dz = eval(params['dz_thin'])
    nside = params['nside']
    N_samp = params['N_sim']
    
    # Defining the bins
    zb  = glass.shells.redshift_grid(0., zmax, dz=dz) # array of all edges of redshift shells
    Nbins = len(zb)-1

    # Load galaxy mask
    gmask = np.load(mask_path)
    gmask = gmask[:Nbins]

    # Load binned BS distance posterior
    bs = np.load(f'{lvk_maps_path}/bs_count_map.npy')
    bs_pdf = np.sum(bs, axis=1).flatten()

    warn_existing_sims(bdir, N_samp, prefix="bs_count_map")

    for k in tqdm(range(N_samp)[rank::size], disable=rank != 0):
        
        gal_count_map = np.load(f'{gdir}/gal_count_map{k+1}.npy')
        gal_count_map[:Nbins]
        bs_map = np.zeros((Nbins,hp.nside2npix(nside)))

        # Choose hosts from the masked galaxy map
        bin_bright_siren = np.random.choice(np.arange(Nbins), p=bs_pdf)
        Bh_pix = sample_hosts(gal_count_map[bin_bright_siren]*gmask[bin_bright_siren].astype(int),
                             1, nside)
            
        bs_map[:,Bh_pix] = bs_pdf[:,None]

        print("Fraction of event in masked region:", 1-np.sum(bs_map*gmask[0]))

        os.makedirs(os.path.dirname(f'{bdir}/bs_count_map{k+1}.npy'), exist_ok=True)
        np.save(f'{bdir}/bs_count_map{k+1}', bs_map)

def count_map_subset(params, rand=False):
    """Builds the total count map for a subset of GW events
        for which sky maps are contained in the lvk_maps directory"""
    params = with_defaults(params) # default values of the optional parameters

    storage = params["storage"]
    lvk_maps_path = storage+params["lvk_maps"]
    bdir_orig = storage+params["bdir"]
    N_sim = params['N_sim']

    if rand==True:
        bdir = bdir_orig+'_rand'

    event_list = []

    for f in os.listdir(lvk_maps_path):

        h5_key = os.path.splitext(f)[0]
        event_list.append(h5_key)


    for k in range(N_sim):

        bbh_count_map = []

        for ev in event_list:
            indv_map = np.load(f'{bdir}/sim{k+1}/{ev}.npy')
            bbh_count_map.append(indv_map)

        bbh_count_map = np.array(bbh_count_map)
        final_map = np.sum(bbh_count_map, axis=0)

        if rand==True:
            os.makedirs(os.path.dirname(f'{bdir_orig}_subset/bbh_count_map{k+1}.npy'), exist_ok=True)
            np.save(f'{bdir_orig}_subset/bbh_count_map{k+1}', final_map)
        else:         
            os.makedirs(os.path.dirname(f'{bdir_orig}_subset_rand/bbh_count_map{k+1}.npy'), exist_ok=True)
            np.save(f'{bdir_orig}_subset_rand/bbh_count_map{k+1}', final_map)

