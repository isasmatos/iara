from iara.generate_matter_cls import matter_power
from iara.pseudo_cls import pseudo_cls, cl_means
from iara.simulate_galaxies import sim_gal, selection_Poisson_dndz, compute_gls
from iara.simulate_gws import sim_gws
from iara.simulate_gws_forecasts import sim_bbh
from iara.mcmc import mcmc, plot_contours

import yaml
from pathlib import Path
import sys
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = MPI.COMM_WORLD.Get_rank() # index of the task

project_root = Path(__file__).resolve().parents[2]


def input_path(argv):
    """Path of the input file.

    --run file.yaml:   path relative to the current folder (or absolute); if it does not exist there,
                       the file of that name in the inputs folder.
    --input file.yaml: file in the inputs folder (or absolute path).
    Default: inputs/params_dls.yaml.
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


def main(argv=None):

    # Importing input parameters -----------------------------------

    params_file = input_path(sys.argv[1:] if argv is None else argv)
    if not params_file.exists():
        raise FileNotFoundError(f"Input file {params_file} not found")

    with open(params_file) as stream:
        params = yaml.safe_load(stream)


    # ----- Compute matter Cls -------------------------------------

    if params["compute_matter_cls"] == True:
        if rank == 0:

            print("Computing matter Cls...")
            cls_matter = matter_power(params)
            cls_matter.compute()

            print("Computing Gaussian Cls...")
            compute_gls(params, monopole=False)

    # Synchronize all ranks: wait until everyone is done
    comm.Barrier()


    # ----- Simulate galaxy maps ---------------------------------

    if params["simulate_galaxies"] == True:

        print("Simulating galaxies...")
        sim_gal(params)
        sim_gal(params, rand=True)

        if params["Nphotoz"] is not None or params["completeness"] is not None:

            print("Adding photo-z's and/or selection function")
            selection_Poisson_dndz(params)
            selection_Poisson_dndz(params, rand=True)


    # ----- Simulate GW positions ---------------------------------

    if params["simulate_gws"] == True:

        if params["forecast"] == True:
            sim_bbh(params)
            sim_bbh(params, rand=True)
        else:
            sim_gws(params)
            sim_gws(params, rand=True)


    # ----- Computing pseudo-Cls -----------------------------------

    if params["compute_cls"] == True:

        pseudo_cls(params)
        pseudo_cls(params, rand=True)

        # Synchronize all ranks: wait until everyone is done
        comm.Barrier()
        # Only one rank computes the mean
        if rank == 0:

            print("Computing mean Cls...")
            cl_means(params)
            cl_means(params, rand=True)


    # ----- MCMC ---------------------------------------------------

    if params["mcmc"] == True:
        if rank == 0:

            mcmc(params)
            plot_contours(params)


if __name__ == "__main__":
    main()
