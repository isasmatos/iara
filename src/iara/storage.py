"""Estimate of the disk space written by a run.

:func:`estimate_storage` counts the files that each step of the pipeline writes (with the shapes
and data types used by the code) and :func:`storage_summary` turns the estimate into the text
printed by :mod:`iara.main` before the run starts.
"""

import os
import numpy as np


def _n_bins(params):
    """Number of redshift bins (bin_edges is a list or the path to a .npy file)."""
    bin_edges = params["bin_edges"]
    if not isinstance(bin_edges, list):
        bin_edges = np.load(bin_edges)
    return len(bin_edges) - 1


def _n_lvk_events(params):
    """Number of LVK sky maps in lvk_maps, or None if the folder is not found."""
    path = params["storage"] + params.get("lvk_maps", "")
    if not os.path.isdir(path):
        return None
    return sum('GW' in f for f in os.listdir(path))


def estimate_storage(params):
    """Disk space, in bytes, that each step set to True in params will write.

    Returns a list of (step, bytes, description) and a list of notes about what is not included.
    The maps are float64 arrays of shape (Nbins, 12 nside^2): each simulation writes a few of them,
    and they are by far the largest outputs. Every simulation step also writes the randoms."""

    Nbins = _n_bins(params)
    npix = 12*params["nside"]**2
    N_sim = params["N_sim"]
    map_size = Nbins*npix*8 # one map of all bins, float64
    rows, notes = [], []

    def maps(n_per_sim):
        return f"{N_sim} sims x 2 (with randoms) x {n_per_sim} map{'s' if n_per_sim > 1 else ''} of {map_size/1e6:.1f} MB"

    if params["compute_matter_cls"]:
        lmax = params["lmax"]
        size = 2*Nbins*(Nbins+1)//2*(lmax+1)*8 # matter Cls and Gaussian Cls of all pairs of bins
        rows.append(("matter Cls", size, "matter and Gaussian Cls"))

    if params["simulate_galaxies"]:
        n = 1 + (params["include_lensing"] == True) + (params["bgw"] != 1)
        rows.append(("galaxy simulations", 2*N_sim*n*map_size, maps(n)))
        if params["completeness"] is not None or params["Nphotoz"] is not None:
            rows.append(("selection function", 2*N_sim*map_size, maps(1) + " (_obs)"))

    if params["simulate_gws"]:
        if params["forecast"]:
            rows.append(("GW simulations", 2*N_sim*2*map_size, maps(2) + " (GW counts and hosts)"))
            if params.get("forecast_save_samples"):
                notes.append("the samples of every event (forecast_save_samples: True) are not included")
        else:
            Nev = _n_lvk_events(params)
            if Nev is None:
                rows.append(("GW simulations", 2*N_sim*map_size, maps(1)))
                notes.append("the maps of each LVK event, one per event per simulation, are not included "
                             "(lvk_maps not found)")
            else:
                rows.append(("GW simulations", 2*N_sim*(1 + Nev)*map_size,
                             maps(1 + Nev) + f" (GW counts and the map of each of the {Nev} events)"))

    if params["compute_cls"]:
        n_bands = (params["lmax"] - 1)//params["bs"] # bands of bs multipoles from ell = 2
        cl_size = n_bands*Nbins**2*4 # float32
        size = 2*(N_sim + 1)*3*cl_size # gg, bb and gb Cls of each simulation and their means
        rows.append(("pseudo-Cls", size, f"{N_sim} sims x 2 (with randoms) x 3 spectra of {cl_size/1e3:.0f} kB"))

    if params["mcmc"]:
        ndim = len(params.get("priors") or {})
        # chain and log-probability, float64, plus a margin: emcee grows the hdf5 file step by step,
        # which makes it several times larger than the data (8.6 MB for 384 kB of data in the tutorial)
        size = params.get("mcmc_steps", 0)*params.get("mcmc_walkers", 0)*(ndim + 1)*8 + 10e6
        rows.append(("MCMC chains", size, "chains (with a margin for the hdf5 overhead) and contour plot"))

    return rows, notes


def _format(size):
    for unit, scale in [("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("kB", 1e3)]:
        if size >= scale:
            return f"{size/scale:.1f} {unit}"
    return f"{size:.0f} B"


def storage_summary(params):
    """Text with the estimated disk space of the run, step by step."""

    rows, notes = estimate_storage(params)
    total = sum(size for _, size, _ in rows)
    width = max([len(step) for step, _, _ in rows], default=0)

    lines = [f"Estimated disk space written by this run: {_format(total)} (1 GB = 10^9 bytes)"]
    for step, size, description in rows:
        lines.append(f"  {step:<{width}}  {_format(size):>9}  {description}")
    for note in notes:
        lines.append(f"  Note: {note}.")
    return "\n".join(lines)
