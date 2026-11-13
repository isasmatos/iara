Input parameters
================

This page lists the parameters of the input YAML file. Use ``inputs/params_dls.yaml``, or the
yaml files of the tutorial in ``examples/tutorial_forecast/inputs``, as templates.

A yaml file only needs the parameters used by the steps it runs. Parameters with a default value
(column "Default") can be left out; the others are required by the steps that use them. The
defaults are defined in ``iara/defaults.py`` and are also applied when the functions of iara are
called directly, e.g. from a notebook. A parameter given as ``null`` keeps the value ``null``.

Steps to run
------------

Set each step to ``True`` to run it. Steps that are not listed are not run.

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``compute_matter_cls``
     - ``False``
     - Compute the matter power spectra and the Gaussian spectra.
   * - ``simulate_galaxies``
     - ``False``
     - Simulate the galaxy maps and their randoms. Also applies the selection function and photometric
       redshifts when ``completeness`` or ``Nphotoz`` is set.
   * - ``simulate_gws``
     - ``False``
     - Simulate the GW maps and their randoms. Uses the forecast module when ``forecast`` is ``True``,
       and the LVK posterior samples otherwise.
   * - ``compute_cls``
     - ``False``
     - Compute the pseudo-spectra of each simulation, inverted with namaster, and their mean over simulations.
   * - ``mcmc``
     - ``False``
     - Run the MCMC and plot the contours.

Simulation settings
-------------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``nside``
     - required
     - HEALPix resolution of the maps.
   * - ``lmax``
     - required
     - Maximum multipole of the matter spectra and of the pseudo-spectra (at most ``3*nside-1``).
   * - ``bin_edges``
     - required
     - Redshift bin edges. Either a list or the full path to a ``.npy`` file.
   * - ``N_sim``
     - required
     - Number of simulations.
   * - ``N_samples``
     - required (forecasts)
     - Number of posterior samples (distance and sky position) of each GW event.
   * - ``bs``
     - required (spectra, MCMC)
     - Integer number of multipoles in one bin of the pseudo-spectra.
   * - ``corr``
     - ``null``
     - Number of correlated matter shells. ``null`` correlates all shells.
   * - ``bgw``
     - ``1``
     - GW bias w.r.t. the galaxies. If it differs from 1, a second set of host galaxy catalogues is
       generated, and the GW hosts are drawn from it. It is also the fiducial GW bias of the likelihood:
       the theory cross-spectrum is rescaled by the sampled ``bgw`` divided by this value.
   * - ``include_lensing``
     - ``False``
     - Whether to include the lensing magnification of the GW distances. The galaxy simulations then also
       save the convergence maps of each redshift bin, ``kappa_map{k}.npy``, which the forecast GW
       simulations read from ``gdir``. The same value must be used in both steps.
   * - ``multiple_surveys``
     - ``False``
     - Whether the galaxies come from several surveys, each with its own files (see `Galaxy data files`_).

Fiducial cosmology
------------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``H0``
     - required
     - Hubble constant, in km/s/Mpc.
   * - ``omch2``
     - required
     - Physical cold dark matter density.
   * - ``ombh2``
     - required
     - Physical baryon density.
   * - ``OM``
     - ``(ombh2+omch2)/h^2``
     - Total matter density. Used in the lensing magnification of the forecasts and as the true value
       in the contour plot. The fiducial ``OM`` of the likelihood is always ``(ombh2+omch2)/h^2``.
   * - ``As``
     - required
     - Amplitude of the primordial power spectrum.
   * - ``ns``
     - required
     - Scalar spectral index.
   * - ``tau``
     - required
     - Optical depth to reionization.
   * - ``w0``, ``wa``
     - ``-1``, ``0``
     - Dark energy equation of state :math:`w(a) = w_0 + w_a(1-a)` (PPF) of the likelihood. They stay
       fixed unless they are in ``priors``.

Paths of the simulations
------------------------

All paths are strings appended to ``storage``, so they start with ``/``.

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``storage``
     - required
     - Main storage folder. ``"."`` is the folder iara is run from.
   * - ``matter_cls``
     - required (matter spectra, galaxies)
     - Matter power spectra and Gaussian spectra.
   * - ``gdir``
     - required (galaxies, GWs, spectra, MCMC)
     - Galaxy simulations. The randoms are in ``gdir`` + ``_rand``, the maps with a selection function in
       ``gdir`` + ``_obs`` and the biased host catalogues in ``gdir`` + ``_bias``. The likelihood reads the
       galaxy maps to get the number of galaxies of each bin.
   * - ``bdir``
     - required (GWs, spectra)
     - GW simulations. The randoms are in ``bdir`` + ``_rand``.
   * - ``cls_dir``
     - required (spectra)
     - Pseudo-spectra of each simulation and their means. The randoms are in ``cls_dir`` + ``_rand``.

Galaxy data files
-----------------

``gal_dir`` is relative to ``storage``, and the other files are relative to ``gal_dir``. For several
surveys (``multiple_surveys: True``), ``dndz``, ``dndz_photoz``, ``footprint_dict`` and ``galaxy_bias``
are dictionaries with one entry for each survey.

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``gal_dir``
     - ``""``
     - Folder with the galaxy data files. Only needed when one of the files below is used.
   * - ``dndz``
     - ``null``
     - Galaxy redshift distribution (number of galaxies in each bin). ``null`` uses a Smail distribution.
   * - ``smail``
     - ``[0.7, 0.8, 2.0, 6.0e+7]``
     - ``[z_mode, alpha, beta, Ngal]`` of the Smail distribution, where ``Ngal`` is the total number of
       galaxies in the sky. Only used when ``dndz`` is ``null``.
   * - ``mask``
     - ``null``
     - Galaxy mask file, of shape ``(npix,)`` or ``(Nbins, npix)``. ``null`` uses a simple galactic cut
       (:math:`|\sin b| > 1/3`, 2/3 of the sky). Used in the pseudo-spectra and for the sky fraction of the
       likelihood.
   * - ``completeness``
     - ``null``
     - Completeness file. ``null`` means no selection function.
   * - ``Nphotoz``
     - ``null``
     - File with the number of photometric galaxies. ``null`` means no photometric redshifts. When
       ``completeness`` or ``Nphotoz`` is set, the spectra and the likelihood use the maps in ``gdir`` + ``_obs``.
   * - ``dndz_photoz``
     - required with ``completeness`` or ``Nphotoz``
     - Photometric redshift kernels of each survey.
   * - ``footprint_dict``
     - required with ``multiple_surveys`` or a selection function
     - Survey footprints.
   * - ``galaxy_bias``
     - required with ``multiple_surveys``
     - Galaxy bias files.
   * - ``survey_dict``
     - required with ``multiple_surveys`` (LVK GWs)
     - YAML file listing the surveys of each GW event.
   * - ``photoz``, ``gal_map``
     - required by ``selection_Poisson``
     - Redshift uncertainty and observed galaxy map, used by ``selection_Poisson``. The pipeline of
       ``main.py`` does not call this function.

GW simulations
--------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``forecast``
     - required (GWs, MCMC)
     - ``True`` for a forecast, ``False`` for real data. In the GW simulations, it chooses between the
       forecast catalogue and the LVK posterior samples. In the likelihood, it chooses the default
       covariance and the data spectra (see `Likelihood and MCMC`_).
   * - ``detector_forecast``
     - required (forecasts)
     - Forecast catalogue, relative to ``storage`` or as an absolute path. Its columns are ``z`` and/or
       ``Dl``, ``ra``, ``dec``, ``err_Dl`` and ``err_area90``.
   * - ``forecast_distance_unit``
     - ``"Mpc"``
     - Unit of ``Dl`` and ``err_Dl``: ``"Mpc"`` or ``"Gpc"``.
   * - ``forecast_sky_selection``
     - ``"ra_averaged"``
     - Sky selection of the events: ``"ra_averaged"`` (only the declinations of the catalogue are used),
       ``"full"`` or ``"none"``.
   * - ``forecast_sel_dz``
     - ``0.5``
     - Thickness of the redshift bins of the selection maps.
   * - ``forecast_sel_nside``
     - ``8``
     - ``nside`` of the event histograms of the selection maps.
   * - ``forecast_sel_smoothing_deg``
     - ``10.0``
     - Gaussian smoothing of the selection maps, in degrees.
   * - ``forecast_err_nside``
     - ``4``
     - Maximum ``nside`` of the sky pixels used to assign errors.
   * - ``forecast_save_samples``
     - ``False``
     - Whether to save the ``(Dl, lat, lon)`` samples of every event.
   * - ``lvk_maps``
     - required (LVK GWs)
     - LVK sky maps, relative to ``storage``.
   * - ``lvk_samples``
     - required (LVK GWs)
     - LVK posterior samples, relative to ``storage``.
   * - ``zmax``, ``dz_thin``
     - required by ``insert_bright_siren``
     - Maximum redshift and thickness of the redshift shells of the bright sirens. The pipeline of
       ``main.py`` does not call this function.

Pseudo-spectra
--------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``galaxy_map_type``
     - ``"contrast"``
     - Galaxy maps of the spectra: ``"contrast"`` (density contrast) or ``"counts"`` (number counts).
       Used by the pseudo-spectra and the likelihood, which must use the same value.

Likelihood and MCMC
-------------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``cls_theory_dir``
     - required
     - Folder with the mean spectra used as theory, relative to ``storage`` (the ``cls_dir`` of the
       spectra step). The mean spectra of the randoms are read from ``cls_theory_dir`` + ``_rand``.
   * - ``cls_data_dir``
     - required
     - Spectra used as data, relative to ``storage``. For forecasts, the folder with the mean spectra.
       For real data (``forecast: False``), the folder with ``cl_gb_data.npy``, or the ``.npy`` file itself.
   * - ``covariance_type``
     - ``null``
     - ``"Fisher"`` for the Gaussian covariance from the mean spectra, or ``"sample"`` for the covariance of
       the simulated spectra. The sample covariance needs ``N_sim`` larger than the number of data points
       plus 2 (Hartlap factor). ``null`` uses ``"Fisher"`` for forecasts and ``"sample"`` otherwise.
   * - ``fsky``
     - ``null``
     - Sky fraction of the Fisher covariance. ``null`` computes it from the galaxy mask; for a
       redshift-dependent mask, this is the mean over the galaxy bins used. Not used with the sample covariance.
   * - ``lbins``
     - required
     - First and last+1 multipole bin used in the likelihood. The bins are those of the pseudo-spectra:
       ``bs`` multipoles each, from :math:`\ell = 2` to ``lmax``.
   * - ``coarse_bin``
     - ``1``
     - Number of simulation bins combined into each likelihood bin. GW counts are summed; galaxy density
       contrasts are averaged with the number of galaxies of each bin as weights, and counts are summed.
   * - ``cut_bins``
     - ``[0, 0, 0]``
     - Number of likelihood bins to remove at ``[low distance, high distance, high redshift]``.
   * - ``priors``
     - required
     - A dictionary of uniform priors with "parameter name": ``[min, max]``. The parameters that can be
       varied are ``H0``, ``OM``, ``bgw``, ``w0`` and ``wa``. Parameters not listed stay fixed at their
       fiducial values. Models with ``w0 + wa >= 0`` are always excluded.
   * - ``chains_path``
     - required
     - Output file of the chains (``.h5``), relative to ``storage``. The contour plot is saved as
       ``chains_path`` + ``_contour_plot.pdf``.
   * - ``mcmc_steps``
     - required
     - Number of steps of each walker.
   * - ``mcmc_walkers``
     - required
     - Number of walkers (at least twice the number of varied parameters).

The number of galaxies in each redshift bin, used to combine and move the galaxy bins, is the mean over
the ``N_sim`` galaxy simulations (``gdir``, or ``gdir`` + ``_obs`` with a selection function), inside the
galaxy mask.
