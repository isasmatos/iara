# Compute matter cls with camb and store it

import numpy as np
import camb
import glass.shells
import glass.ext.camb
import warnings
import os, os.path

warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")


class matter_power(object):
    ''' This class computes the angular power spectrum of matter density fluctuations
    using CAMB and stores it in a specified directory.
    It takes cosmological parameters and redshift shells as input. '''

    def __init__(self, params):
        
        # Storage
        self.storage = params['storage']
        self.output_dir = self.storage + params["matter_cls"]

        # Cosmology
        self.H0 = params['H0']
        self.omch2 = params['omch2']
        self.ombh2 = params['ombh2']
        self.h = self.H0/100
        self.As = params['As']
        self.ns = params['ns']
        self.tau = params['tau']

        # Simulation set up
        self.lmax = params['lmax']
        self.bin_edges = params['bin_edges']


    def _set_camb_params(self):
        """Set up CAMB parameters."""

        pars = camb.set_params(H0=self.H0, omch2=self.omch2, ombh2=self.ombh2,
                            As=self.As, ns=self.ns, tau=self.tau,
                            NonLinear=camb.model.NonLinear_both)

        # --- Add the Accuracy Control ---
        pars.Transfer.redshift_step = 0.0005   # smaller step for narrow bins
        pars.Transfer.l_switch_limber = 100    # start Limber approx. later

        return pars
    

    def _make_windows(self):
        """Construct the redshift window functions from arbitraly spaced bin edges."""

        if type(self.bin_edges) != list:
            zb = np.load(self.bin_edges)
        else:
            zb = np.array(self.bin_edges)

        dz_arr = zb[1:]-zb[:-1]
        # window function
        ws = glass.shells.tophat_windows(zb, dz=np.min(dz_arr)/100) 

        return ws
    

    def compute(self):
        """Compute matter Cls with camb and store it."""

        warnings.warn(f"Check CAMB accuracy parameters are sufficient for your bin size when using very thin bins. Limber apx. used for l > 100.", category=UserWarning, stacklevel=2)

        pars = self._set_camb_params()
        ws = self._make_windows()

        os.makedirs(os.path.dirname(f'{self.output_dir}/cls.npy'), exist_ok=True)

        cls = glass.ext.camb.matter_cls(pars, self.lmax, ws)
        np.save(f'{self.output_dir}/cls', cls) # matrix with Nmultipoles*Nzbins*(Nzbins+1)/2 elements
