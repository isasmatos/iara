"""Default values of the optional input parameters.

Every public function or class of iara that receives the parameters of the input yaml
applies these defaults first (with_defaults), so a yaml only needs the parameters used
by the steps it runs. The list is also documented in docs/parameters.rst.
"""

import copy


DEFAULTS = {
    # Steps to run
    "compute_matter_cls": False,
    "simulate_galaxies": False,
    "simulate_gws": False,
    "compute_cls": False,
    "mcmc": False,

    # Simulation settings
    "corr": None,               # number of correlated matter shells (None = all shells are correlated)
    "bgw": 1,                   # GW bias (also the fiducial GW bias of the likelihood)
    "include_lensing": False,
    "multiple_surveys": False,

    # Fiducial cosmology (OM is derived from H0, ombh2 and omch2 when it is not given)
    "w0": -1.,                  # dark energy equation of state w(a) = w0 + wa*(1-a)
    "wa": 0.,

    # Galaxy survey
    "gal_dir": "",              # folder of the galaxy data files (mask, dndz, completeness, ...)
    "dndz": None,               # None = Smail distribution
    "smail": [0.7, 0.8, 2.0, 6.0e+7],  # (z_mode, alpha, beta, Ngal) of the Smail distribution
    "mask": None,               # None = simple galactic cut (2/3 of the sky)
    "completeness": None,       # None = no selection function
    "Nphotoz": None,            # None = no photometric redshifts

    # GW forecasts
    "forecast_distance_unit": "Mpc",
    "forecast_sky_selection": "ra_averaged",
    "forecast_sel_dz": 0.5,
    "forecast_sel_nside": 8,
    "forecast_sel_smoothing_deg": 10.0,
    "forecast_err_nside": 4,
    "forecast_save_samples": False,

    # Cls
    "galaxy_map_type": "contrast",

    # Likelihood
    "cut_bins": [0, 0, 0],
    "coarse_bin": 1,
    "covariance_type": None,    # None = "Fisher" for forecasts, "sample" otherwise
    "fsky": None,               # None = sky fraction of the galaxy mask
}


def with_defaults(params):
    """Copy of params with the default value of every optional parameter that is missing.
    Parameters given in params (even as null) are kept. If OM is missing, it is derived
    as (ombh2 + omch2)/h^2, the same value used as fiducial in the likelihood."""

    full = copy.deepcopy(DEFAULTS)
    full.update(params)

    if "OM" not in params and all(p in params for p in ["H0", "ombh2", "omch2"]):
        full["OM"] = (params["ombh2"] + params["omch2"])/(params["H0"]/100)**2

    return full
