"""Command line entry point of IARA: ``iara --run file.yaml``.

Reads the input yaml, applies the default values of the optional parameters
(:mod:`iara.defaults`) and runs the steps that are set to ``True``: matter Cls, galaxy
simulations, GW simulations, pseudo-Cls and MCMC.
"""

from iara.generate_matter_cls import matter_power
from iara.pseudo_cls import pseudo_cls, cl_means
from iara.simulate_galaxies import sim_gal, selection_Poisson_dndz, compute_gls
from iara.simulate_gws import sim_gws
from iara.simulate_gws_forecasts import sim_bbh
from iara.mcmc import mcmc, plot_contours
from iara.defaults import with_defaults
from iara.storage import storage_summary

import yaml
from pathlib import Path
import sys
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = MPI.COMM_WORLD.Get_rank() # index of the task

project_root = Path(__file__).resolve().parents[2]


def input_path(argv):
    """Path of the input file.

    - ``--run file.yaml``: path relative to the current folder (or absolute); if it does not exist
      there, the file of that name in the inputs folder.
    - ``--input file.yaml``: file in the inputs folder (or absolute path).
    - Default: inputs/params_dls.yaml.
    """
    for flag in ["--run", "--input"]:
        if flag in argv:
            i = argv.index(flag)
            if i + 1 >= len(argv):
                raise ValueError(f"Expected a filename after {flag}")
            name = Path(argv[i + 1]).expanduser()
            if flag == "--run" and name.exists():
                return name.resolve()
            return project_root / "inputs" / name
    return project_root / "inputs" / "params_dls.yaml"


def log(message):
    """Print a message once (from rank 0), not once per MPI process."""
    if rank == 0:
        print(message, flush=True)


def main(argv=None):

    # Importing input parameters -----------------------------------

    params_file = input_path(sys.argv[1:] if argv is None else argv)
    if not params_file.exists():
        raise FileNotFoundError(f"Input file {params_file} not found")

    with open(params_file) as stream:
        params = yaml.safe_load(stream)

    # Print the settings of the input file
    log(f"Settings from {params_file}:\n" + yaml.safe_dump(params, sort_keys=False, default_flow_style=None))

    # Optional parameters missing in the input file take their default values (see iara/defaults.py)
    params = with_defaults(params)

    # Estimate of the disk space that the run will write
    log(storage_summary(params) + "\n")

    # ----- Compute matter Cls -------------------------------------

    if params["compute_matter_cls"] == True:
        if rank == 0:

            log("Computing matter Cls...")
            cls_matter = matter_power(params)
            cls_matter.compute()

            log("Computing Gaussian Cls...")
            compute_gls(params, monopole=False)

    # Synchronize all ranks: wait until everyone is done
    comm.Barrier()


    # ----- Simulate galaxy maps ---------------------------------

    if params["simulate_galaxies"] == True:

        log("Simulating galaxies...")
        sim_gal(params)
        log("Simulating galaxy randoms...")
        sim_gal(params, rand=True)

        if params["Nphotoz"] is not None or params["completeness"] is not None:

            log("Adding photo-z's and/or selection function")
            selection_Poisson_dndz(params)
            selection_Poisson_dndz(params, rand=True)


    # ----- Simulate GW positions ---------------------------------

    if params["simulate_gws"] == True:

        if params["forecast"] == True:
            log("Simulating GWs...")
            sim_bbh(params)
            log("Simulating GW randoms...")
            sim_bbh(params, rand=True)
        else:
            log("Simulating GWs...")
            sim_gws(params)
            log("Simulating GW randoms...")
            sim_gws(params, rand=True)


    # ----- Computing pseudo-Cls -----------------------------------

    if params["compute_cls"] == True:

        log("Computing Cls...")
        pseudo_cls(params)
        log("Computing Cls for randoms...")
        pseudo_cls(params, rand=True)

        # Synchronize all ranks: wait until everyone is done
        comm.Barrier()
        # Only one rank computes the mean
        if rank == 0:

            log("Computing mean Cls...")
            cl_means(params)
            cl_means(params, rand=True)


    # ----- MCMC ---------------------------------------------------

    if params["mcmc"] == True:
        if rank == 0:

            log("Running MCMC...")
            mcmc(params)
            plot_contours(params)


if __name__ == "__main__":
    main()
