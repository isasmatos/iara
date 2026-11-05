Input parameters
================

This page lists the parameters of the input YAML file. Use ``inputs/params_dls.yaml`` as a template.

Steps to run
------------

Set each step to ``True`` to run it or ``False`` to skip it.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Parameter
     - Description
   * - ``compute_matter_cls``
     - Compute the matter power spectra and the Gaussian spectra.
   * - ``simulate_galaxies``
     - Simulate the galaxy maps and their randoms. Also applies the selection when ``Nphotoz`` or ``completeness`` is set.
   * - ``simulate_gws``
     - Simulate the GW maps and their randoms. Uses the forecast module when ``forecast`` is ``True``.
   * - ``compute_cls``
     - Compute the pseudo-spectra, invert with namaster, and compute the mean over simulations.
   * - ``mcmc``
     - Run the MCMC.

Simulation settings
-------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Parameter
     - Description
   * - ``lmax``
     - Maximum multipole.
   * - ``bin_edges``
     - Redshift bin edges. Either a list or the full path to a ``.npy`` file.
   * - ``nside``
     - HEALPix resolution of the maps.
   * - ``N_sim``
     - Number of simulations.
   * - ``N_samples``
     - Number of distance samples for each GW event. Used in forecasts only.
   * - ``corr``
     - Number of correlated matter shells. Use ``null`` to correlate all shells.
   * - ``bs``
     - Integer number of multipoles in one bin.
   * - ``bgw``
     - GW bias w.r.t the galaxies. If it differs from 1, a second set of host galaxy catalogues is generated.
   * - ``gw_bias``
     - GW bias for the forecast module. See ``bgw``.
   * - ``include_lensing``
     - Whether to include lensing.
   * - ``multiple_surveys``
     - Whether the galaxies come from several surveys, each with its own files.

Fiducial cosmology
------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Parameter
     - Description
   * - ``H0``
     - Hubble constant.
   * - ``omch2``
     - Physical cold dark matter density.
   * - ``ombh2``
     - Physical baryon density.
   * - ``OM``
     - Total matter density.
   * - ``As``
     - Amplitude of the primordial power spectrum.
   * - ``ns``
     - Scalar spectral index.
   * - ``tau``
     - Optical depth to reionization.

Forecasts
---------

All ``forecast_*`` parameters are optional. The defaults are given below.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Parameter
     - Description
   * - ``forecast``
     - ``True`` for a forecast, ``False`` for real data.
   * - ``detector_forecast``
     - Forecast catalogue, relative to ``storage`` or as an absolute path. Its columns are ``z`` and/or ``Dl``, ``ra``, ``dec``, ``err_Dl`` and ``err_area90``.
   * - ``forecast_distance_unit``
     - Unit of ``Dl`` and ``err_Dl``: ``"Mpc"`` (default) or ``"Gpc"``.
   * - ``forecast_sky_selection``
     - Sky selection: ``"ra_averaged"`` (default), ``"full"`` or ``"none"``.
   * - ``forecast_sel_dz``
     - Thickness of the redshift bins of the selection maps. Default: 0.5.
   * - ``forecast_sel_nside``
     - ``nside`` of the event histograms of the selection maps. Default: 8.
   * - ``forecast_sel_smoothing_deg``
     - Gaussian smoothing of the selection maps, in degrees. Default: 10.0.
   * - ``forecast_err_nside``
     - Maximum ``nside`` of the sky pixels used to assign errors. Default: 4.
   * - ``forecast_save_samples``
     - Whether to save the samples of every event. Default: ``False``.

Simulation paths
----------------

These paths are relative to ``storage``.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Parameter
     - Description
   * - ``storage``
     - Main storage folder.
   * - ``matter_cls``
     - Matter power spectra.
   * - ``bdir``
     - GW simulations.
   * - ``gdir``
     - Galaxy simulations.
   * - ``cls_dir``
     - Computed spectra.

Galaxy data files
-----------------

``gal_dir`` is relative to ``storage``, and the other paths are relative to ``gal_dir``. For several surveys, ``dndz``, ``dndz_photoz``, ``footprint_dict`` and ``galaxy_bias`` are dictionaries with one entry for each survey.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Parameter
     - Description
   * - ``gal_dir``
     - Folder with the galaxy data.
   * - ``mask``
     - Mask file. Use ``null`` for a simple galactic cut.
   * - ``completeness``
     - Completeness file. Use ``null`` for no selection function.
   * - ``Nphotoz``
     - File with the number of photometric galaxies. Use ``null`` if not needed.
   * - ``photoz``
     - Redshift uncertainty, used by ``selection_Poisson``. The default pipeline in ``main.py`` does not call this function.
   * - ``gal_map``
     - Observed galaxy map, used by ``selection_Poisson``.
   * - ``dndz``
     - Galaxy redshift distribution. Use ``null`` for a Smail distribution.
   * - ``dndz_photoz``
     - Photometric redshift kernels.
   * - ``survey_dict``
     - YAML file listing the surveys for each GW event.
   * - ``footprint_dict``
     - Survey footprints.
   * - ``galaxy_bias``
     - Galaxy bias files.

GW data files
-------------

These paths are relative to ``storage``.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Parameter
     - Description
   * - ``lvk_maps``
     - LVK sky maps.
   * - ``lvk_samples``
     - LVK posterior samples.

Likelihood and MCMC
-------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Parameter
     - Description
   * - ``chains_path``
     - Output file for the chains, relative to ``storage``.
   * - ``coarse_bin``
     - Number of simulation bins combined into each likelihood bin.
   * - ``cut_bins``
     - Number of bins to remove at ``[low distance, high distance, high redshift]``.
   * - ``lbins``
     - First and last multipole bin used in the likelihood.
   * - ``cls_theory_dir``
     - Folder with the theory spectra, relative to ``storage``.
   * - ``cls_data_dir``
     - Data spectra, relative to ``storage``.
   * - ``n_gal``
     - Full path to a text file with the number of galaxies per redshift bin. Use ``null`` to compute it from the simulated maps.
   * - ``priors``
     - A dictionary of uniform priors with "parameter name": ``[min, max]``, for example for ``H0``, ``OM`` and ``bgw``. Parameters not listed stay fixed at their fiducial values.
