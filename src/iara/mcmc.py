# MCMC for cosmological parameters estimation

import os
from getdist import plots, MCSamples
import numpy as np
import matplotlib.pyplot as plt
import emcee
import iara.likelihood as lk


def mcmc(params):
       

    # Initializing likelihood
    par = lk.global_parameters(params)
    print("Initialized global parameters")
    theory = lk.cl_theory(par)
    print("Initialized theory")
    data = lk.cl_data(par)
    print("Initialized data")

    if par.forecast == True:
        inv_cov = lk.inverse_covariance(par, theory)
        like = lk.likelihood(par, inv_cov, theory, data)
        print("Initialized likelihood for forecasts")
    else:
        sample_cov = lk.sample_covariance(par)
        print("Initialized sample covariance")
        like = lk.likelihood_real_data(par, sample_cov, theory, data)
        print("Initialized likelihood for real data")

    filepath = f'{params["storage"]}{params["chains_path"]}'
    backend = emcee.backends.HDFBackend(filepath)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Initial positions of the walkers: tiny gaussian ball around the fiducial
    # model, one dimension per varied parameter (fiducial value = par.<NAME>).
    fid = np.array([getattr(par, name) for name in par.varied_params], dtype=float)
    ndim = len(par.varied_params)
    nwalkers = params["mcmc_walkers"]
    pos = fid + 1e-2 * fid * np.random.randn(nwalkers, ndim)


    backend.reset(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, like.log_posterior,
        backend=backend,
        moves=[
            (emcee.moves.DESnookerMove(), 0.1),
            (emcee.moves.DEMove(), 0.9 * 0.9),
            (emcee.moves.DEMove(gamma0=1.0), 0.9 * 0.1)],
        #args=[par.gw_bias], # extra arguments of log_posterior
        # Differential Evolution Markov Chain with snooker updater and fewer chains
        # https://link.springer.com/content/pdf/10.1007/s11222-008-9104-9.pdf
        # DE snooker paper suggests DE 90% and DE snooker 10% of the time  (section 3)
        # and with a 10% chance to set gamma to 1.0 to allow for ocasional large jumps
        )

    max_n = params["mcmc_steps"]

    # We'll track how the average autocorrelation time estimate changes
    index = 0
    autocorr = np.empty(max_n)
    
    # This will be useful to testing convergence
    old_tau = np.inf
    
    for sample in sampler.sample(pos, iterations=max_n, progress=True):
        # Only check convergence every 100 steps
        if sampler.iteration % 100:
            continue
        
        # Compute the autocorrelation time so far
        # Using tol=0 means that we'll always get an estimate even
        # if it isn't trustworthy
        tau = sampler.get_autocorr_time(tol=0)
        autocorr[index] = np.mean(tau)
        index += 1
    

        # Check convergence
        converged = np.all(tau * 100 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        #if converged: # both conditions satisfied
        #   break
        old_tau = tau

    
def plot_contours(params, param_names=None, param_labels=None):
    """Plot contours from a single chain."""
    
    filepath1 = f'{params["storage"]}{params["chains_path"]}'
    reader1 = emcee.backends.HDFBackend(filepath1, read_only=True) 
    sample1 = reader1.get_chain(flat=True)
    # Remove burn-in (5%)
    sample1 = sample1[int(len(sample1)*0.05):]
    print("Number of samples:", len(sample1))

    ranges = params["priors"].copy()

    if param_names is None:
        param_names = list(ranges.keys())

    if param_labels is None:
        param_labels = param_names

    for par in param_names:
        ranges[par] = tuple(ranges[par])

    print("Parameter names:", param_names)
    print("Ranges:", ranges)
    print("Fiducial values:", {
        par: params[par] for par in param_names
    })
        
    samples1 = MCSamples(samples=sample1,names = param_names, labels=param_labels, ranges=ranges)

    truths = tuple([params[param_names[p]] for p in range(len(param_names))])
    samples1.updateSettings({'smooth_scale_2D': 2.})
    #samples1.updateSettings({'smooth_scale_1D': 6.})

    g = plots.get_single_plotter(ratio=1,width_inch=9)
    g.settings.title_limit_fontsize = 24
    g.settings.axes_labelsize=24
    g.settings.axes_fontsize=14
    g.settings.legend_fontsize=24
    g.settings.axis_marker_color = 'black'
    g.settings.axis_marker_lw=1
    g.settings.axis_tick_x_rotation=45
    g.settings.linewidth = 3
    g.settings
    g.triangle_plot([samples1],
                    filled=True, 
                    #legend_labels=
                    legend_loc='upper right',
                    title_limit=False, # first title limit (for 1D plots) is 68% by default
                    markers=dict(zip(param_names, truths)),
                    alphas = [1.0, 0.8,1.0],
                    ranges=ranges)#,contour_args={'smooth_scale_2D': 3.});

    g.fig.tight_layout(rect=[0, 0, 1, 0.95])
    g.fig.set_constrained_layout(True)
    plt.savefig(f"{params['storage']}{params['chains_path']}_contour_plot.pdf")

    # Print results of marginalized 1D posteriors
    print('Marginalized 1D Best Fits +- 1sigma')
    for p in param_names:
        # marginalized mode (peak of density)
        marg = samples1.get1DDensity(p)
        mode_val = marg.x[marg.P.argmax()]

        # parameter stats from summary
        low, high = marg.getLimits(0.68)[0], marg.getLimits(0.68)[1]

        print(f"{p}: {mode_val:.3f} (+{high - mode_val:.3f}, -{mode_val - low:.3f})")

    
def plot_various_contours(params, chain_files, chain_labels, param_names, param_labels):
    """Overplot contours from different chains."""

    ranges = params["priors"].copy()

    for par in param_names:
        ranges[par] = tuple(ranges[par])

    readers = []
    samples = []
    n = 0
    for filepath in chain_files:

        print(filepath)

        readers.append(emcee.backends.HDFBackend(filepath, read_only=True)) 
        samp = readers[n].get_chain(flat=True)
        # Remove burn-in (5%)
        samp = samp[int(len(samp)*0.05):]
        samples.append(MCSamples(samples=samp, names = param_names, labels=param_labels, ranges=ranges))

        n += 1
   
    truths = tuple([params[param_names[p]] for p in range(len(param_names))])
    
    for samp in samples:
        samp.updateSettings({'smooth_scale_2D': 2.})
    #samples2.updateSettings({'smooth_scale_2D': 2.})
    #samples2.updateSettings({'smooth_scale_1D': 6.})

    g = plots.get_single_plotter(ratio=1,width_inch=9)
    g.settings.title_limit_fontsize = 24
    g.settings.axes_labelsize=24
    g.settings.axes_fontsize=14
    g.settings.legend_fontsize=24
    g.settings.axis_marker_color = 'black'
    g.settings.axis_marker_lw=1
    g.settings.axis_tick_x_rotation=45
    g.settings.linewidth = 3
    g.settings
    g.triangle_plot(samples,
                    filled=True, 
                    legend_labels=chain_labels,
                    legend_loc='upper right',
                    title_limit=None, # first title limit (for 1D plots) is 68% by default
                    markers=dict(zip(param_names, truths)),
                    alphas = [1.0, 0.8,1.0])#,contour_args={'smooth_scale_2D': 3.});

    g.fig.tight_layout(rect=[0, 0, 1, 0.95])
    g.fig.set_constrained_layout(True)
    plt.savefig(f"{params['storage']}_contour_plot.pdf")
    
