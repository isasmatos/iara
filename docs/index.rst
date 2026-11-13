.. iara documentation master file, created by
   sphinx-quickstart on Tue Nov  3 15:37:04 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

IARA documentation
==================

IARA: *Inference from Angular cross-correlations for Redshift-peak Analysis*

IARA is a code made for cosmological parameter inference from the angular cross-correlations of localized gravitational wave-sources and galaxies.

It implements the Peak Sirens method, where both large-scale structure tracers are binned into thin radial tomographic bins, and the peak of their cross-spectrum reveals the Hubble diagram. 

The analysis is currently based on the construction of thousands of light-cones from GLASS lognormal mocks. Errors on GW source position and distances are included, as well as non-Gaussian GW posteriors, galaxy photometric redshifts, selection functions and mask.

IARA is flexible enough to be used in very different regimes: the many GW sources regime of 3g detectors and the poorly localized few source regime of current data.

Citing IARA
----------------
If you use IARA in your research, please cite the following papers:

- `Ferri et al. JCAP 04 (2025) 008
  <https://iopscience.iop.org/article/10.1088/1475-7516/2025/04/008>`_

- `Santiago de Matos et al. (2026) ApJ 1011 175
  <https://iopscience.iop.org/article/10.3847/1538-4357/ae9a7a>`_

- `Dalang et al. (2026) arXiv:2610.32436
  <https://arxiv.org/abs/2610.32436>`_

Getting started
---------------


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage
   parameters
   api
