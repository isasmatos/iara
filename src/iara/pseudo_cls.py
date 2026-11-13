# Compute pseudo-Cls of galaxy and GW simulations

from time import time
import numpy as np
import healpy as hp
import pymaster as nmt
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
from scipy.special import legendre
from scipy.integrate import quad
import os, os.path
from mpi4py import MPI


rank = MPI.COMM_WORLD.Get_rank() # index of the task
size = MPI.COMM_WORLD.Get_size() # total number of tasks
barrier = MPI.COMM_WORLD.Barrier


def galaxy_mask(params, Nbins):
    """Galaxy mask used in the pseudo-Cls (and for the sky fraction in the likelihood).
    Returns the mask file in gal_dir, of shape (npix,) or (Nbins, npix), or, if "mask" is null,
    a simple galactic cut |sin(b)| > 1/3 for all bins, of shape (Nbins, npix)."""

    if params['mask'] is None:

        # Simple galactic cut mask
        print("Using simple galactic cut mask")
        f_sky_masked = 1/3
        nside = params['nside']
        npix = hp.nside2npix(nside)
        lon, lat = hp.pix2ang(nside, np.arange(npix), lonlat=True)
        vis = np.where(abs(lat) < np.arcsin(f_sky_masked)*180/np.pi, 0., 1.)
        gmask = np.zeros(shape=(Nbins, npix))
        gmask[:] = vis

    else:
        mask_path = params["storage"]+params["gal_dir"]+params["mask"]
        print(mask_path)
        gmask = np.load(mask_path)

    return gmask


def pseudo_cls(params, rand=False):
    
    """Computes pseudo-cls given galaxy and bbh catalogs.
    For galaxies, masks 1/3 of the sky. Uses bandpowers."""

    storage = params["storage"]
    gdir = storage+params["gdir"]
    bdir = storage+params["bdir"]
    cls_dir = storage+params["cls_dir"]
    forecast = params["forecast"]
    galaxy_map_type = params.get("galaxy_map_type", "contrast") # galaxy maps: "contrast" or "counts"
    if galaxy_map_type not in ['contrast', 'counts']:
        raise ValueError(f"galaxy_map_type must be 'contrast' or 'counts', not {galaxy_map_type}")

    if rand == True:
        gdir += '_rand'
        bdir += '_rand'
        cls_dir += '_rand'
        print('Using random galaxy maps')

    if params['completeness'] is not None or params['Nphotoz'] is not None:
        gdir += '_obs' # with photo-z uncertainties or selection function applied
        print('Using observed galaxy maps')
         
    lmax = params['lmax']
    nside = params['nside']
    N_samp = params['N_sim']
    bs = params['bs']
    bin_edges = params['bin_edges']

    if type(bin_edges) != list:
        bin_edges = np.load(bin_edges)
    else:
        bin_edges = np.array(bin_edges)
 
    # Binning
    zb = bin_edges
    npix = hp.nside2npix(nside)
    Nbins = len(zb)-1

    # Pre-computing the coupling matrices
    lmax_cls = lmax
    b = nmt.NmtBin.from_lmax_linear(lmax_cls, bs) # binning in bandpowers, starts from ell=2 and removes last bandpowers if bin is incomplete
    n_bands = b.get_n_bands()

    # Get redshift dependent full galaxy mask
    gmask = galaxy_mask(params, Nbins)

    bmask=nmt.NmtField(np.ones(npix), None, spin=0, lmax=lmax_cls, masked_on_input=False)

    # Computing coupling matrices only once for all sims
    w_bb = nmt.NmtWorkspace()
    w_bb.compute_coupling_matrix(bmask, bmask, b)
    w_gg = np.empty(Nbins, dtype='object')
    w_gb = np.empty(Nbins, dtype='object')

    # Save time if mask is the same for all bins, i.e. shape(gmask) = (Npixels)
    redshift_independent_mask = (gmask.ndim == 1)
    print("redshift_independent_mask:", redshift_independent_mask)

    for i in range(Nbins):

        mask_data = gmask if redshift_independent_mask else gmask[i, :]
        gmask_i = nmt.NmtField(mask_data, None, spin=0, lmax=lmax, masked_on_input=False)
        w_gg[i] = nmt.NmtWorkspace() # objects to store mode-coupling matrices
        w_gb[i] = nmt.NmtWorkspace()
        w_gg[i].compute_coupling_matrix(gmask_i, gmask_i, b)
        w_gb[i].compute_coupling_matrix(gmask_i, bmask, b)

    print("Computed coupling matrices")

    # Computing pseudo-cls
    for k in tqdm(range(N_samp)[rank::size]):

        Pcls_gb = np.zeros(shape=(n_bands, Nbins, Nbins), dtype='float32')
        Pcls_gg = np.zeros(shape=(n_bands, Nbins, Nbins), dtype='float32')
        Pcls_bb = np.zeros(shape=(n_bands, Nbins, Nbins), dtype='float32')

        alm_g = np.zeros(shape=(Nbins, np.sum(np.arange(lmax_cls+1)+1)), dtype='complex')
        alm_b = np.zeros(shape=(Nbins, np.sum(np.arange(lmax_cls+1)+1)), dtype='complex')

        # Get galaxy count map from simulations
        # For null test change k+1 to k+2 here
        gal_count_map = np.load(f'{gdir}/gal_count_map{k+1}.npy')
        gal_count_map = gal_count_map[:Nbins]

        if forecast == True:

            # Get bbh count map from simulations
            bh_count_map = np.load(f'{bdir}/bbh_count_map{k+1}.npy')
            bh_count_map = bh_count_map[:Nbins]

        else:

            # Get bbh count map from simulations
            bh_count_map = np.load(f'{bdir}/bbh_count_map{k+1}.npy')
            bh_count_map = bh_count_map[:Nbins]
            
        start=time()
        for i in range(Nbins):

            mask_data = gmask if redshift_independent_mask else gmask[i, :]
            map_masked = gal_count_map[i] * mask_data
            # Density contrast of the galaxies, or number counts with galaxy_map_type: "counts"
            if galaxy_map_type == 'contrast':
                ngal = np.average(map_masked[mask_data>0])
                map_masked = mask_data * (map_masked - ngal)/ngal

            alm_g[i] = hp.map2alm(map_masked,lmax=lmax_cls)
            alm_b[i] = hp.map2alm(bh_count_map[i], lmax=lmax_cls) 
          
        # Healpix assumes cls are symmetric, but the cross-cls are not
        upper_cls_gb = hp.alm2cl(alm_g,alm_b, lmax=lmax_cls) # shape (N*(N+1)/2, 200)
        lower_cls_gb = hp.alm2cl(alm_b,alm_g, lmax=lmax_cls)
        cls_gg = hp.alm2cl(alm_g, lmax=lmax_cls)
        cls_bb = hp.alm2cl(alm_b, lmax=lmax_cls)

        # Inverting the pseudo-cls            
        idx = 0
        for n in range(Nbins):
            for i in range(Nbins - n):
                j = n + i
                Pcls_gb[:,i,j] = w_gb[i].decouple_cell(upper_cls_gb[idx][np.newaxis,:], cl_bias=None)
                Pcls_gb[:,j,i] = w_gb[j].decouple_cell(lower_cls_gb[idx][np.newaxis,:], cl_bias=None)
                Pcls_gg[:,i,j] = Pcls_gg[:,j,i] = w_gg[i].decouple_cell(cls_gg[idx][np.newaxis,:], cl_bias=None)
                Pcls_bb[:,i,j] = Pcls_bb[:,j,i] = w_bb.decouple_cell(cls_bb[idx][np.newaxis,:], cl_bias=None)
                idx += 1
            
        print(f'saving cls of simulation {k+1}')
        os.makedirs(os.path.dirname(f'{cls_dir}/Cls_gg{k+1}.npy'), exist_ok=True)
        os.makedirs(os.path.dirname(f'{cls_dir}/Cls_bb{k+1}.npy'), exist_ok=True)
        os.makedirs(os.path.dirname(f'{cls_dir}/Cls_gb{k+1}.npy'), exist_ok=True)
        np.save(f'{cls_dir}/Cls_gg{k+1}', Pcls_gg)
        np.save(f'{cls_dir}/Cls_bb{k+1}', Pcls_bb)
        np.save(f'{cls_dir}/Cls_gb{k+1}', Pcls_gb)
        
        
def cl_means(params, rand=False):
    """Compute the average Cls over N_samp simulations"""
    
    storage = params["storage"]
    cls_dir = storage+params["cls_dir"]

    if rand == True:
        cls_dir += '_rand'
        print('Using random cls')

    lmax = params['lmax']
    N_samp = params['N_sim']
    bs = params['bs']
    bin_edges = params['bin_edges']

    if type(bin_edges) != list:
        bin_edges = np.load(bin_edges)
    else:
        bin_edges = np.array(bin_edges)

    Nbins = len(bin_edges)-1
    
    lmax_cls = lmax#-1
    b = nmt.NmtBin.from_lmax_linear(lmax_cls, bs) # binning in bandpowers, starts from ell=2 and removes last bandpowers if bin is incomplete
    lbmax=b.get_effective_ells().size

    sims = os.listdir(cls_dir)
    for file in sims:
        if 'Cls_' not in file:
            sims.remove(file)

    N = int(len(sims)/3)   
    print(f'{N} simulations')
    
    res_gg = np.zeros(shape=(lbmax, Nbins, Nbins),dtype='float32')
    res_bb = np.zeros(shape=(lbmax, Nbins, Nbins),dtype='float32')
    res_gb = np.zeros(shape=(lbmax, Nbins, Nbins),dtype='float32')
    
    for i in tqdm(range(N_samp)):
        if f'Cls_gg{i+1}.npy' in sims:
            res_gg += np.real(np.load(f'{cls_dir}/Cls_gg{i+1}.npy'))/N
            res_bb += np.real(np.load(f'{cls_dir}/Cls_bb{i+1}.npy'))/N
            res_gb += np.real(np.load(f'{cls_dir}/Cls_gb{i+1}.npy'))/N
        else:
            continue

    os.makedirs(os.path.dirname(f'{cls_dir}/Cl_gg_mean.npy'), exist_ok=True)        
    os.makedirs(os.path.dirname(f'{cls_dir}/Cl_gb_mean.npy'), exist_ok=True)        
    os.makedirs(os.path.dirname(f'{cls_dir}/Cl_bb_mean.npy'), exist_ok=True)        
    np.save(f'{cls_dir}/Cl_gg_mean.npy', res_gg)
    np.save(f'{cls_dir}/Cl_gb_mean.npy', res_gb)
    np.save(f'{cls_dir}/Cl_bb_mean.npy', res_bb)


def legendre_integral(theta_min, theta_max, l_max, lbvec):
    """Computes the legendre integrals (2l+1)Pl(cos(theta)), which gives the weights for each Cl summed.
       Theta_min and theta_max in degrees. Returns array with shape (1,lmax)."""
    
    idx = np.digitize(l_max, lbvec)
    ls = lbvec[:idx+1]
    results = np.zeros(len(ls))
    for i, l in enumerate(ls):
        results[i], _ = quad(lambda x: legendre(int(l))(np.cos(x)), np.deg2rad(theta_min), np.deg2rad(theta_max))

    return results/(theta_max-theta_min) * (2*ls + 1)/(4*np.pi)


def cl_gb_lsum(params,cl,l_max,theta_vec):
  """Computes the matrix which is the sum of all Cl's, from l=0 to l=l_max(theta),
     with weights (2l+1) Pl(cos(theta)) integrated in theta."""

  lmax = params['lmax']
  lmax_cls = lmax-1
  bs = params['bs']
  b = nmt.NmtBin.from_lmax_linear(lmax_cls, bs) # binning in bandpowers, starts from ell=2 and removes last bandpowers if bin is incomplete
  lbvec=b.get_effective_ells()
  
  if l_max > lbvec[-1]:
      l_max = lbvec[-1]
      print("Too high l_max. Using l_max=%s instead."%l_max)
 
  cls_lsum=np.zeros(shape=(len(theta_vec)-1,cl.shape[1],cl.shape[2]),dtype='float32')
  idx0 = np.digitize(l_max, lbvec)
  Pl_integrals=np.zeros(shape=(len(theta_vec)-1, len(lbvec[:idx0+1])))

  for t in tqdm(range(len(theta_vec)-1)):
    Pl_integrals[t] = legendre_integral(theta_vec[t], theta_vec[t+1], l_max, lbvec)

  for t in range(len(theta_vec)-1):
    if len(np.where(np.sign(Pl_integrals[t])<0)[0])==0:
        # if for no l the Legendre integral is negative, get indice of l_max
      idx=np.digitize(l_max, lbvec)
    else:
        # if that happens for some l, stop in first l with negative contribution
      idx=int(np.where(np.sign(Pl_integrals[t])<0)[0][0])
      print(idx)
      
    for j in range(idx):
        cls_lsum[t,:,:] += cl[j,:,:] * Pl_integrals[t,j]

  return cls_lsum
    
