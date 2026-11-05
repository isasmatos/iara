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

# Importing input parameters -----------------------------------

project_root = Path(__file__).resolve().parents[2]

# Default input file
input_file = "params_dls.yaml"

# Allow: --input other_file.yaml
if "--input" in sys.argv:
    i = sys.argv.index("--input")
    if i + 1 >= len(sys.argv):
        raise ValueError("Expected a filename after --input")
    input_file = sys.argv[i + 1]

params_file = project_root / "inputs" / input_file

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

