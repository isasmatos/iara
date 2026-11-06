## IARA

<p align="center">
  <img src="docs/logo.png" alt="iara logo" width="200">
</p>

IARA (*Inference from Angular cross-correlations for Redshift-peak Analysis*) is a code for cosmological parameter inference from the angular cross-correlations of localized gravitational-wave sources and galaxies. It implements the Peak Sirens method.

### Installation

1. Create a new conda environment and install the required Python packages:

    - python=3.11.7
    - numpy=2.2.4
    - scipy
    - pyyaml
    - pandas
    - astropy
    - healpy
    - h5py
    - matplotlib
    - namaster=2.4
    - glass==2025.2
    - glass.ext.camb
    - camb
    - cosmology.compat.camb
    - transformcl
    - emcee
    - mpi4py
    - tqdm
    - arviz
    - getdist

    <!-- Not imported by the code: glass.camb, lal -->

2. From the main folder of the repository, install IARA:

    > pip install -e .

### Usage

1. Create an input file in the `inputs` folder, using one of the existing `.yaml` files as a template. You may have multiple input files for different scenarios you want to analyse.

2. Edit the parameters of your input file according to the setting you want. Every parameter is described in `docs/parameters.rst`. There are a few main sets of parameters there:

    - Parameters of simulations and/or data, like bin size, nside, etc
    - Paths to the relevant directories where to store your simulations
    - Paths of where your data is located, including names of files like '/path/completeness_kband.npy'
    - Which modules you would like to run: `compute_matter_cls`, `simulate_galaxies`, `simulate_gws`, `compute_cls` and `mcmc`. Example: set `simulate_galaxies` to True and everything else to False if you want to just simulate the galaxies, after having computed the matter power spectrum.
    - Whether you are running a forecast or analysing real data, as those two are currently handled a bit differently in some modules.

3. Run:

    > iara --run <your_input_file>.yaml

    or, for parallelization,

    > mpirun -np <ntasks> iara --run <your_input_file>.yaml

    With `--run`, the input file is a path relative to the folder you run from, or an absolute path; if no such file exists, IARA looks for a file of that name in the `inputs` folder. The `iara` command is installed by `pip install -e .` (reinstall if you installed IARA before this command existed).

    The older form still works:

    > python -m iara.main --input <your_input_file>.yaml

    With `--input`, give either the name of a file in the `inputs` folder or an absolute path. Without `--run` or `--input`, IARA uses `inputs/params_dls.yaml`. These commands need the editable install from the installation step, because IARA finds the `inputs` folder relative to the location of its source code.


### Summary of code structure

The code contains 8 modules, in `src/iara`:


generate_matter_cls.py

    This module computes the 3D matter power spectrum in harmonic space in the fiducial cosmology. One needs to specify the redshift shells for this calculation and the multipole range.


simulate_galaxies.py

    This module contains several functions to simulate galaxy maps from a given matter power spectrum. It uses the lognormal code GLASS. One needs to specify the average number of galaxies desired per redshift shell.
    
    Optionally one can input a selection function (completeness) and photometric redshift uncertainties. In this case, a second set of 'observed' maps will be saved.

    By default, a set of randoms is generated, with galaxy maps draw from a zero power spectrum dark matter field.


simulate_gws.py

    This module builds GW maps on top of the galaxy maps. One needs to give a set of 3D sky maps, which will be moved around the sky in each simulation according to the position of the new host galaxies. It does not include lensing magnification.

    If the value of the input GW bias is different from 1, another set of galaxies is produced from the same dark matter field, but with a different bias. These will be the host galaxies from which later GW sources will be generated.

    By default, a set of randoms for GWs is generated from the randoms for galaxies.


simulate_gws_forecasts.py

    This module also produces GW maps, but in this case one needs to provide a file with Fisher forecasts, containing detectable GW events with their 3D positions, and errors. This version contains lensing magnification (optionally) and reproduces the anisotropic sensitivity of GW detectors. A network of GW detectors needs to be specified (ex: 'ET+2CE' or 'LVK').

    Again if the GW bias differs from 1, another set of host galaxies will be generated. Again, randoms are also generated.


pseudo_cls.py

    This module computes the angular power spectra between the GW and galaxy maps. If a mask is given, it uses namaster to deconvolve the spectra and return the "true" underlying spectra. Masks can be redshift dependent. If no mask is given, it assumes a simple mask covering the Milky Way.

likelihood.py

    This module builds the likelihood. It currently contains two classes for likelihoods: "likelihood" for forecasts, being the version of Ferri et al. 2025, and "likelihood_real_data", the version in Santiago de Matos et al. 2026.

        likelihood class: a Gaussian covariance is assumed and the amplitude of the cross-spectrum is not rescaled by the average number of galaxies when rebinning in a new cosmology;

        likelihood_real_data class: a sample covariance is computed from the simulations and the cross-spectrum's amplitude is rescaled by the average number of galaxies when rebinning in a new cosmology.

mcmc.py

    This module runs the MCMC, using emcee.

