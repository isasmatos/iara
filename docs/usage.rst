Usage
=====

Installation
------------

Create a new conda environment and install the required Python packages:

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

.. Not imported by the code: glass.camb, lal

Then, from the main folder of the repository, install IARA:

.. code-block:: bash

   pip install -e .

Setting the input parameters
----------------------------

IARA reads its settings from a YAML file, usually kept in the ``inputs`` folder. You can keep
several input files for the different scenarios you want to analyse. Each file
sets:

- The parameters of the simulations and data, such as bin size and ``nside``.
- The directories where your simulations are stored.
- The location of your data files.
- Which modules to run. For example, to simulate only the galaxies after
  computing the matter power spectrum, set ``simulate_galaxies`` to ``True``.
  Modules that are not listed, or are set to ``False``, are not run.
- Whether you are running a forecast or analysing real data. These are currently
  handled differently in some modules.

A file only needs the parameters of the modules it runs: the optional parameters
that are left out take their default values (see :doc:`parameters`).

Running the code
----------------

Run:

.. code-block:: bash

   iara --run my_params.yaml

To run in parallel, use:

.. code-block:: bash

   mpirun -np <ntasks> iara --run my_params.yaml

With ``--run``, the input file is a path relative to the folder you run from,
or an absolute path. If no such file exists, IARA looks for a file of that name
in the ``inputs`` folder.

The older form still works:

.. code-block:: bash

   python -m iara.main --input my_params.yaml

With ``--input``, give either the name of a file in the ``inputs`` folder or an
absolute path. Without ``--run`` or ``--input``, IARA uses
``inputs/params_dls.yaml``.

These commands need the editable install above, because IARA finds the
``inputs`` folder relative to the location of its source code.

See :doc:`parameters` for a description of every parameter.
