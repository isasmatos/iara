
import numpy as np
import matplotlib.pyplot as plt
import bisect
import pymaster as nmt
import os
import camb
from matplotlib import rcParams
import warnings
from scipy.interpolate import interp1d, CubicSpline


rcParams['mathtext.fontset'] = 'stix'

plt.rc('text', usetex=False)
plt.rc('font', family='serif')


class global_parameters(object):
    ''' This class defines global parameters used in the likelihood analysis.
    '''

    def __init__(self, params):
        
        # Storage
        self.storage = params['storage']

        # Fiducial cosmology
        self.H0 = params['H0']
        self.omch2 = params['omch2']
        self.ombh2 = params['ombh2']
        self.h = self.H0/100
        self.OM = (self.ombh2 + self.omch2)/(self.h**2)
        self.As = params['As']
        self.ns = params['ns']
        self.tau = params['tau']
        self.bgw = params['bgw']

        pars = camb.set_params(H0=self.H0, omch2=self.omch2, ombh2=self.ombh2,As=self.As,ns=self.ns,tau=self.tau,
                            NonLinear=camb.model.NonLinear_both)

        self.camb_back = camb.get_background(pars)

        # There are two potentially different binnigs here:
        # 1. thin bins for the simulations,
        # 2. and maybe thicker bins for likelihood

        # 1) Binning of simulations
        bin_edges = params['bin_edges']

        if type(bin_edges) != list:
            self.zb = np.load(bin_edges)
        else:
            self.zb = np.array(bin_edges)
        
        # luminosity distance bin edges corresponding to the uniform redshift bin edges:
        self.D_zb = self.camb_back.luminosity_distance(self.zb)
        self.Nbins = len(self.zb)-1 # number of bins in the simulations (thin bins)
        self.zmax = self.zb[-1]
        self.dz = self.zb[1]-self.zb[0] # deprecated

        # 2) Binning of likelihood (now necessarily the same as the one of simulations)
        self.cuts = params['cut_bins']
        self.lower_cut, self.upper_cut, self.z_cut = self.cuts
        coarse_factor = params['coarse_bin']
        self.zmin = 0
        # z bin edges for likelihood ---> change this array to change the binning
        self.gal_zs2 = self.zb #np.linspace(self.zmin, self.zmax, self.bh_bins+1) #uniform gal redshift bin edges
        self.gal_bins = len(self.gal_zs2)-1 # number of galaxy bins before cutting
        self.bh_bins = len(self.gal_zs2)-1 # number of bbh bins before cutting
        self.gbins = self.gal_bins-self.z_cut
        self.bbins = self.bh_bins-self.upper_cut-self.lower_cut
        # D bins defined by the fiducial model
        self.fid_D = self.camb_back.luminosity_distance(self.gal_zs2)

        # Parameters for the multipole binning
        self.nside = params['nside']
        self.lmax = params['lmax']
        self.lmax_cls = self.lmax-1
        self.bs = 5 # bandwidth used for pseudo-cls
        b = nmt.NmtBin.from_lmax_linear(self.lmax_cls, self.bs)
        self.lbvec = b.get_effective_ells()
        self.lbmax = b.get_effective_ells().size
        self.lb_idx_min = params['lbins'][0]
        self.lb_idx_max = params['lbins'][1]

        # Other parameters of the MCMC
        self.N_sim = params['N_sim'] # number of simulations
        self.fsky = params['fsky']
        self.forecast = params["forecast"]
        self.cls_th_dir = f'{self.storage}{params["cls_theory_dir"]}' # path to theory cls
        self.cls_data_dir = f'{self.storage}{params["cls_data_dir"]}' # path to data cls
        self.cls_th_dir_rand = f'{self.storage}{params["cls_theory_dir"]}_rand' # path to cls of randoms
                
        # which parameters to vary in the MCMC
        self.priors = params['priors']
        self.varied_params = list(self.priors.keys())

        if params['n_gal'] is not None:
            self.ng = np.genfromtxt(params['n_gal']) # average number of galaxies in each bin, for fiducial Cl
        else:
            warnings.warn("Loading n_gal from simulated maps. If you have a selection function, consider providing n_gal path in the .yaml input file" ,category=UserWarning, stacklevel=2)
            g_map = np.load(f'{self.storage}{params["gdir"]}/gal_count_map1.npy')
            self.ng = np.sum(g_map, axis=1) # sum over pixels in each redshift bin


    def mask_h0_prior(self, cl_gb, h0min=None, h0max=None):
        """Multiplies the matrix cl_gb by a prior in h0
        cl_gb : cross cls binned in the likelhood binning
                already with cuts in z and dl bins"""
        
        if h0min is None:
            h0min = self.h0_min
        if h0max is None:
            h0max = self.h0_max

        # effective bins of BBHs after cuts
        if self.upper_cut > 0:
            bins_b = self.gal_zs2[self.lower_cut:-self.upper_cut]
        else:
            bins_b = self.gal_zs2[self.lower_cut:]

        # effective bins of BBHs after cuts
        if self.upper_cut > 0:
            bins_b = self.gal_zs2[self.lower_cut:-self.upper_cut]
        else:
            bins_b = self.gal_zs2[self.lower_cut:]

        bins_b_mean = (bins_b[1:]+bins_b[:-1])/2
        dl_bins_means = self.camb_back.luminosity_distance(bins_b_mean)
        
        # effective bins of galaxies
        bins_g = np.linspace(self.zmin, self.zmax, self.gal_bins+1)
        bins_g_mean = (bins_g[1:]+bins_g[:-1])/2
        if self.z_cut > 0:
            bins_g_mean = bins_g_mean[:-self.z_cut]
        else:
            bins_g_mean = bins_g_mean

        # Let's find the value of H0 for each bin with fixed OM=0.3   
        H0_bins = np.zeros((self.gbins, self.bbins))

        for i in range(self.gbins):
            for j in range(self.bbins):
                H0_bins[i,j] = self.H0*self.camb_back.luminosity_distance(bins_g_mean[i])/dl_bins_means[j]
        
        prior = np.zeros((self.gbins, self.bbins))
        prior = np.where((H0_bins<h0max) & (H0_bins>h0min), 1, 0)
        
        return prior*cl_gb[:]

    def stack_Cls(self, cl_gb, cl_gg, cl_bb):
        
        """Build a matrix containing both auto and cross-correlations.
        Excludes bins we don't want & apply H0 prior mask to Cl_gb."""
        
        full_cl = np.zeros((self.lb_idx_max-self.lb_idx_min, self.gbins+self.bbins, self.gbins+self.bbins), dtype='float32')
        full_cl[:, :self.gbins, :self.gbins] = cl_gg[self.lb_idx_min:self.lb_idx_max,:self.gbins,:self.gbins]
        full_cl[:, self.gbins:, self.gbins:] = cl_bb[self.lb_idx_min:self.lb_idx_max,self.lower_cut:self.bbins+self.lower_cut,self.lower_cut:self.bbins+self.lower_cut]
        #cl_bb[:,self.lower_cut:self.bbins-self.upper_cut,self.lower_cut:self.bbins-self.upper_cut]
        full_cl[:, :self.gbins, self.gbins:] = cl_gb[self.lb_idx_min:self.lb_idx_max,:self.gbins,self.lower_cut:self.bbins+self.lower_cut]#cl_gb[:,:self.gbins,self.lower_cut:self.bbins-self.upper_cut])
        full_cl[:, self.gbins:, :self.gbins] = np.transpose(cl_gb[self.lb_idx_min:self.lb_idx_max,:self.gbins,self.lower_cut:self.bbins+self.lower_cut], (0, 2, 1))

        #print("Shape of fulll Cl [0]:", full_cl.shape[0])
        return full_cl
    

    def rebinning_Cl(self, CL, type='zz'):

        """This function is used to rebin Cl matrices without cosmology dependence,
        according to the binning we want for the likelihood anlaysis.
        
        Args:
            CL ('array') : The original 3D C_l(z,z') matrix from where you'll obtain another C_l. 
            
            type ('string') : the possible types of C_l are: 'zz' or 'DfidDfid'.
                    'zz' corresponds to a reshaped version of your original C_l(z,z') into a matrix of shape
                    (lmax,gal_bins,gal_bins). 
                    'DfidDfid' corresponds to a C_l(D,D) where D are luminosity distance bins equivalent to (zmin,zmax)
                    in your fiducial model, and the transformation z->D is done in the fiducial cosmology.
        """

        new_D = self.D_zb
        gal_idx = np.array([bisect.bisect_right(self.zb, z)-1 for z in self.gal_zs2]) # how many thin bins fit in thick bins

        if type=='zz':
            matrix_bin2 = np.zeros(shape=(CL.shape[0], self.gal_bins, self.gal_bins), dtype = 'float32')     # final matrix after z and z' binning
            matrix_bin1 = np.zeros(shape=(CL.shape[0], CL.shape[1], self.gal_bins), dtype = 'float32')  # matrix after z' binning
            sub = 0
            for i in range(1,len(gal_idx)-1):
                exceed = (-self.zb[gal_idx[i]] + self.gal_zs2[i])/(self.zb[gal_idx[i]+1]-self.zb[gal_idx[i]]) * CL[:,:,gal_idx[i]]
                matrix_bin1[:,:,i-1] = np.sum(CL[:,:,gal_idx[i-1]:gal_idx[i]], axis=2) + exceed - sub
                sub = exceed
            matrix_bin1[:,:,i] = np.sum(CL[:,:,gal_idx[i]:gal_idx[i+1]],axis=2)
            sub = 0
            for i in range(1,len(gal_idx)-1):
                exceed = (-self.zb[gal_idx[i]]+self.gal_zs2[i])/(self.zb[gal_idx[i]+1]-self.zb[gal_idx[i]]) * matrix_bin1[:,gal_idx[i],:]
                matrix_bin2[:,i-1,:] = np.sum(matrix_bin1[:,gal_idx[i-1]:gal_idx[i],:],axis=1) + exceed - sub
                sub = exceed
            matrix_bin2[:,i,:] = np.sum(matrix_bin1[:,gal_idx[i]:gal_idx[i+1],:],axis=1)

            return np.real(matrix_bin2)

        if type=='DfidDfid':
            D_idx = np.array([bisect.bisect_right(new_D, d)-1 for d in self.fid_D])#how many thin new D bins fit inside the fiducial cosmology 'bh_bins' D bins
            if np.max(D_idx)==(len(self.zb)-1):
                D_idx=D_idx[D_idx<(len(self.zb)-1)]
                D_idx=np.append(D_idx,[(len(self.zb)-1)])
            else:
                D_idx=D_idx[D_idx<(len(self.zb)-1)]
            matrix_bin1 = np.zeros(shape=(CL.shape[0],CL.shape[1], self.bh_bins), dtype = 'float32')  # matrix after fiducial D binning
            matrix_bin2 = np.zeros(shape=(CL.shape[0],self.bh_bins, self.bh_bins), dtype = 'float32') # matrix after z binning
            sub = 0
            for i in range(1,len(D_idx)-1):
                exceed = ((self.fid_D[i]-new_D[D_idx[i]] )/(new_D[D_idx[i]+1]-new_D[D_idx[i]])) * CL[:,:,D_idx[i]]
                matrix_bin1[:,:,i-1] = np.sum(CL[:,:,D_idx[i-1]:D_idx[i]], axis=2) + exceed - sub
                sub = exceed
            matrix_bin1[:,:,i] = np.sum(CL[:,:,D_idx[i]:D_idx[i+1]], axis=2)
            sub = 0
            for i in range(1,len(D_idx)-1):
                exceed = ((self.fid_D[i]-new_D[D_idx[i]] )/(new_D[D_idx[i]+1]-new_D[D_idx[i]])) * matrix_bin1[:,D_idx[i],:]
                matrix_bin2[:,i-1,:] = np.sum(matrix_bin1[:,D_idx[i-1]:D_idx[i],:],axis=1) + exceed - sub
                sub = exceed
            matrix_bin2[:,i,:] = np.sum(matrix_bin1[:,D_idx[i]:D_idx[i+1],:],axis=1)

            return np.real(matrix_bin2)
        
        if type=='zDfid':

            #### new distance bin edges ####
            D_idx = np.array([bisect.bisect_right(new_D, d)-1 for d in self.fid_D])#how many thin new D bins fit inside the fiducial cosmology 'bh_bins' D bins
            if np.max(D_idx)==(len(self.zb)-1):
                D_idx=D_idx[D_idx<(len(self.zb)-1)]
                D_idx=np.append(D_idx,[(len(self.zb)-1)])
            else:
                D_idx=D_idx[D_idx<(len(self.zb)-1)]
            matrix_bin1 = np.zeros(shape=(CL.shape[0],CL.shape[1], self.bh_bins), dtype = 'float32')  # matrix after fiducial D binning
            matrix_bin2 = np.zeros(shape=(CL.shape[0],self.gal_bins, self.bh_bins), dtype = 'float32') # matrix after z binning
            sub = 0
            # Rebinning for BBH bins
            for i in range(1,len(D_idx)-1):
                exceed = ((self.fid_D[i]-new_D[D_idx[i]] )/(new_D[D_idx[i]+1]-new_D[D_idx[i]])) * CL[:,:,D_idx[i]]
                matrix_bin1[:,:,i-1] = np.sum(CL[:,:,D_idx[i-1]:D_idx[i]], axis=2) + exceed - sub
                sub = exceed
                #print(new_D[D_idx[i]])
            matrix_bin1[:,:,i] = np.sum(CL[:,:,D_idx[i]:D_idx[i+1]], axis=2)
            sub = 0
            # Rebinning for galaxy bins
            for i in range(1,len(gal_idx)-1):
                exceed = ((-self.zb[gal_idx[i]]+self.gal_zs2[i])/(self.zb[gal_idx[i]+1]-self.zb[gal_idx[i]])) * matrix_bin1[:,gal_idx[i],:]
                matrix_bin2[:,i-1,:] = np.sum(matrix_bin1[:,gal_idx[i-1]:gal_idx[i],:],axis=1) + exceed - sub
                sub = exceed
            matrix_bin2[:,i,:] = np.sum(matrix_bin1[:,gal_idx[i]:gal_idx[i+1],:],axis=1)

            return np.real(matrix_bin2)


class likelihood_real_data(object):

    def __init__(self, par, samp_cov, theory, data, reuse_inv_cov=False):

        self.params = par
        self.theory = theory
        self.Cl_gb_fid = theory.cl_fid
        self.Cl_gb_data = data.data_cls(data_type='data')
        self.cov, self.Nsim_eff = samp_cov.cov()
        print("Computed sample covariance")

        # Check cuts are properly made
        if reuse_inv_cov == True:
            print("Reusing inverse covariance matrix from file")
            self.inv_cov = np.load(f'{self.params.cls_th_dir}/inv_cov.npy')
        else:
            cuts_idx_z = np.r_[:self.params.gbins]
            cuts_idx_dl = np.r_[self.params.lower_cut:self.params.bh_bins-self.params.upper_cut]
            cov_cut = self.cov[np.ix_(np.arange(self.cov.shape[0]),cuts_idx_z,cuts_idx_dl,np.arange(self.cov.shape[0]),cuts_idx_z,cuts_idx_dl)]
            self.inv_cov = samp_cov.compute_inv_cov(cov_cut, self.Nsim_eff)
            np.save(f'{self.params.cls_th_dir}/inv_cov.npy', self.inv_cov)
            print("Inverted covariance matrix")
        #self.cuts_bs = [3,7] # intermediate distance bins to remove (because they have no GWs)
        # Use cuts_bs = [1,1] for incld. all intermediate dl bins

    def chi2(self, h0, om, bgw):
        
        """Chi squared using sample covariance. Version of Santiago de Matos et al. 2025."""

        # New theoretical Cl_gb
        Cl_gb_theory = self.theory.new_Cl(self.Cl_gb_fid, h0, om)
        
        # Cuts in distance and redshift
        Nbins = Cl_gb_theory.shape[1] # Redefining number of bins after rebinning
        
        cuts = np.zeros(4, dtype='int')
        cuts[[0,1,3]] = self.params.cuts
        cuts[2] = 0 # no cut at low z now, but data likelihood is adapted to that
        cuts_idx_z = np.r_[cuts[2]:Nbins-cuts[3]]
        #cuts_idx_dl = np.r_[cuts[0]:self.cuts_bs[0], self.cuts_bs[1]:Nbins-cuts[1]] # for removing intermediate bins
        cuts_idx_dl = np.r_[cuts[0]:Nbins-cuts[1]]

        Cl_gb_theory = bgw*Cl_gb_theory[self.params.lb_idx_min:self.params.lb_idx_max,cuts_idx_z[:, None],cuts_idx_dl]

        # Get cross-spectrum only
        Cl_data = self.Cl_gb_data[:, :self.params.gbins, self.params.gbins:]

        dC = (Cl_gb_theory - Cl_data)
        dC2 = np.copy(dC)
        
        T = np.einsum('lzd,lzdmsp,msp->', dC, self.inv_cov, dC2)

        return T

    def log_like(self, pars, remove_small_scale_residual=False):
        """Log likelihood."""

        h   = pars["H0"]
        om  = pars["OM"]
        bgw = pars["bgw"]
        chi = self.params.bs*self.chi2(h, om, bgw)
        
        if remove_small_scale_residual:
            chi_smallest_ells = np.average(chi[:-15])
            chi -= chi_smallest_ells
            print(f"Removed small scale residual in chi2, using average of 15 last bands.")
        
        return -0.5 * np.sum(chi)


    def log_prior(self, th_args):
        """Uniform priors on all varied parameters.
        For ET+2CE, use for (h0, om), mins = 65, 0.1, maxs = 75, 0.5
        For LVK O5, use mins = 40, 0.05, maxs = 100, 0.95"""

        for name, value in zip(self.params.varied_params, th_args):

            pmin, pmax = self.params.priors[name]
            print(pmin, pmax, value)

            if not (pmin < value < pmax):
                return -np.inf

        return 0.0

    def log_posterior(self, th_args):    
        """Log posterior."""

        pars = {
            "H0": self.params.H0,
            "OM": self.params.OM,
            "bgw": self.params.bgw,
        }

        for name, value in zip(self.params.varied_params, th_args):
            pars[name] = value

         # Log prior
        lp = self.log_prior(th_args)
        
        if not np.isfinite(lp):
            return -np.inf
        
        return lp + self.log_like(pars)


class sample_covariance(object):
    """Computes the covariance matrix from simulations."""

    def __init__(self, par):

        self.params = par

    def sum_every_n_from_start(self, arr, n=4, axis=1, start=8):
        """Take arrays with thin bins and sum every n bins along axis starting from index start.
        Useful if we use different bin sizes in different redshift ranges
        like in Santiago de Matos et al. 2025."""

        # Split before and after start
        before = np.take(arr, range(0, start), axis=axis)
        after = np.take(arr, range(start, arr.shape[axis]), axis=axis)

        # Trim the "after" part to multiple of n
        length = after.shape[axis] // n * n
        slices = [slice(None)] * arr.ndim
        slices[axis] = slice(0, length)
        after = after[tuple(slices)]

        # Reshape and sum groups of n
        new_shape = list(after.shape)
        new_shape[axis] = new_shape[axis] // n
        new_shape.insert(axis + 1, n)
        summed = after.reshape(new_shape).sum(axis=axis + 1)

        # Concatenate back
        result = np.concatenate([before, summed], axis=axis)
        return result

    def cov(self, type='Cls_gb'):
        """Compute covariance of Cls over Nsim simulations.
        Default is to compute only covariance of Cross-Cls.
        Set type = 'Cls_gg' to compute instead covariance of galaxy auto-Cls."""
           
        # Get Cls from each simulation
        sims = os.listdir(self.params.cls_th_dir)

        cell = np.zeros((self.params.N_sim, ((self.params.Nbins)**2)*(self.params.lb_idx_max-self.params.lb_idx_min)))
        print(np.shape(cell))
        print(self.params.N_sim)

        for k in range(self.params.N_sim):

            if type+f'{k+1}.npy' in sims:
                data = np.load(f'{self.params.cls_th_dir}/{type}{k+1}.npy')[self.params.lb_idx_min:self.params.lb_idx_max,:,:] # shape = (Nlb, Nbins_thin, Nbins_thick)
                cell[k] = data.flatten()

        cell = cell[~np.all(cell == 0, axis=1)] # remove rows without any simulation
        Nsim_eff = len(cell)

        cl_cov = np.cov(cell.T) # numpy cov takes a data vector where rows are variables and columns are realizations
        cl_cov = cl_cov.reshape(np.shape(data)*2)
        
        return cl_cov, Nsim_eff

    def compute_inv_cov(self, cov, Nsim, mask=False, Neff=False):
        """Invert covariance matrix with Hartlap correction and optionally apply mask for H0 prior (non-default)."""
        print("Computing inverse covariance matrix")
        if mask==True: # for cov with len(shape) = 6
            mask_matrix = self.params.mask_h0_prior(np.zeros((self.params.lb_idx_max-self.params.lb_idx_min,self.params.gbins,self.params.bbins)), h0min=self.params.h0_min, h0max=self.params.h0_max)
            mask_flat = mask_matrix.reshape(cov.shape[1]*cov.shape[2])
            mask_full = np.tile(mask_flat, cov.shape[0])
            inds = np.nonzero(mask_full)[0] # indices we keep

            # Reshape covariance
            cov_2d_orig = cov.reshape((cov.shape[0]*cov.shape[1]*cov.shape[2],cov.shape[0]*cov.shape[1]*cov.shape[2]))
            # Extract sub-covariance
            cov_2d = cov_2d_orig[np.ix_(inds, inds)]

        else: 
            if len(np.shape(cov))==6:

                cov_2d = cov.reshape((cov.shape[0]*cov.shape[1]*cov.shape[2],cov.shape[0]*cov.shape[1]*cov.shape[2]))
    
            if len(np.shape(cov))==4:

                cov_2d = cov.reshape((cov.shape[0]*cov.shape[1],cov.shape[0]*cov.shape[1]))

        inv_cov_2d = np.linalg.pinv(cov_2d, rcond=1e-16)
        print("Inverted covariance matrix")

        if mask==True:
            # Build full inverse matrix with zeros for masked variables
            Cinv_full = np.zeros_like(cov_2d_orig)
            Cinv_full[np.ix_(inds, inds)] = inv_cov_2d
            inv_cov_2d = np.copy(Cinv_full)

        # Hartlap correction
        Ndof = cov_2d.shape[0] # number of degrees of freedom

        if Nsim < Ndof+3:
            warnings.warn(f'Less simulations than degrees of freedom: problem with Hartlap factor',category=UserWarning, stacklevel=2)
        
        inv_cov_2d = ((Nsim-Ndof-2)/Nsim)*inv_cov_2d

        # Effective DOF
        if Neff==True:
            tr = np.trace(cov_2d)
            tr2 = np.trace(cov_2d @ cov_2d)
            N_eff = (tr**2) / tr2
            print(f'Effective DOF = {N_eff}')

        print("Hartlap correction applied to inverse covariance matrix")

        return inv_cov_2d.reshape(np.shape(cov))



class likelihood(object):

    def __init__(self, par, inv_cov, theory, data):

        self.params = par
        self.theory = theory
        self.Cl_gb_fid = theory.cl_fid
        self.Cl_gb_data = data.data_cls()
        self.inv_corr = inv_cov

        # Computing the inverse correlation matrix and normalization matrix once with fiducial cosmology
        # If you want to vary them with cosmology, they should be computed inside chi2  
        self.fiducial_norm, self.fiducial_inv_corr, self.fiducial_corr, self.Gl_full_nonorm = inv_cov.compute_inv_corr(self.params.bgw, self.params.H0, self.params.OM)

    def chi2(self, h0, om, bgw):
    
        """Chi squared version of Ferri et. al, i.e. assuming Gaussian fields."""
        
        # Create a zero matrix Lx100x100 to use as theoretical Cl_gg and cl_bb components
        gg_mask=np.zeros(shape=(self.params.lbmax,self.params.gal_bins,self.params.gal_bins))
        bb_mask=np.zeros(shape=(self.params.lbmax,self.params.bh_bins,self.params.bh_bins))
        
        # New theoretical Cl_gb
        Cl_gb_theory = bgw*self.theory.new_Cl(self.Cl_gb_fid, h0, om)

        # H0 prior mask can be applied inside stack_Cls
        
        # Theoretical Cl full stack
        Cl_theory_full = self.params.stack_Cls(Cl_gb_theory,gg_mask,bb_mask)


        # filtered l bandpower vector
        lbvec_filtered = self.params.lbvec[self.params.lb_idx_min:self.params.lb_idx_max]
        lbvec_filtered_idx = np.arange(self.params.lb_idx_min,self.params.lb_idx_max,dtype='int')
        
        T1 = np.zeros(len(lbvec_filtered))

        # computing T1 contributions to each \ell in our chi-squared:
        for i in range(len(lbvec_filtered)):
        
            ell=lbvec_filtered[i]
            idx=i #lbvec_filtered_idx[i] # This changed because now we cut the ells before
        
            # C is delta-cl = Cl_data - Cl_theo
            C = np.triu((self.Cl_gb_data[idx] - Cl_theory_full[idx])/self.fiducial_norm[idx])
            # D is the inverse of the fiducial correlation matrix (Gamma^-1)
            D = self.fiducial_inv_corr[idx] 

            CtD = np.dot(C.T,D)
            DtC = CtD.T
            
            CtDC = np.dot(np.dot(C.T,D),C)
            
            CtDCdotD = np.dot(CtDC,D)
            DtCdotDtC = np.dot(DtC,DtC)
            
            t1a = np.trace(CtDCdotD)
            t1b = np.trace(DtCdotDtC)
            T1[i]  = (2*ell+1) * self.params.bs * (t1a + t1b)

        return T1 # Array of chi2(ell)

    def log_like(self, pars, remove_small_scale_residual=False):
        """Log likelihood."""

        h   = pars["H0"]
        om  = pars["OM"]
        bgw = pars["bgw"]
        chi = self.chi2(h, om, bgw)
        
        if remove_small_scale_residual:
            chi_smallest_ells = np.average(chi[:-15])
            chi -= chi_smallest_ells
            print(f"Removed small scale residual in chi2, using average of 15 last bands.")
        
        return -0.5 * np.sum(chi)

    def log_prior(self, th_args):
        """Uniform priors on all varied parameters.
        For ET+2CE, use for (h0, om), mins = 65, 0.1, maxs = 75, 0.5
        For LVK O5, use mins = 40, 0.05, maxs = 100, 0.95"""

        for name, value in zip(self.params.varied_params, th_args):

            pmin, pmax = self.params.priors[name]
            print(pmin, pmax, value)

            if not (pmin < value < pmax):
                return -np.inf

        return 0.0

    def log_posterior(self, th_args):    
        """Log posterior."""

        pars = {
            "H0": self.params.H0,
            "OM": self.params.OM,
            "bgw": self.params.bgw,
        }

        for name, value in zip(self.params.varied_params, th_args):
            pars[name] = value

         # Log prior
        lp = self.log_prior(th_args)
        
        if not np.isfinite(lp):
            return -np.inf
        
        return lp + self.log_like(pars)


class priors(object):
    """In construction..."""

    def spectral_siren_prior_gwtc4(self, h0):
        """H0 prior from the analysis of GWTC-4 with spectral sirens only"""
        #spec_siren_prior = np.genfromtxt(f'{root}/gwtc4-spectral-posterior.csv', delimiter=',')
        spec_siren_prior = np.genfromtxt(f'{root}/gwtc4-spectral-MLTP-posterior.csv', delimiter=',')
        spec_func = interp1d(spec_siren_prior[:,0], spec_siren_prior[:,1])
        return spec_func(h0)

    def bright_siren_prior_gwtc4(self, h0):
        """H0 prior from the analysis of GWTC-4 with spectral sirens + GW170817"""
        bright_siren_prior = np.genfromtxt(f'{root}/gwtc4-bright-posterior.csv', delimiter=',')
        bright_func = interp1d(bright_siren_prior[:,0], bright_siren_prior[:,1])
        return bright_func(h0)


class inverse_covariance(object):

    """Computes the inverse of the Gamma matrix."""

    def __init__(self, par, theory):

        self.params = par
        self.theory = theory

    def inv_corr(self, cl1_corr, rcond):
        """Inverse of Gls"""
        invcorr = np.zeros(cl1_corr.shape)
        for ell in range(cl1_corr.shape[0]):
            try:
                invcorr[ell] = np.linalg.pinv( cl1_corr[ell] , rcond = rcond )
            except(np.linalg.LinAlgError): # exception aadded for now due to problems at low ell
                
                warnings.warn(f'Could not invert Gl matrix for $\ell$-band {ell+self.params.lb_idx_min}',category=UserWarning, stacklevel=2)
                invcorr[ell] = np.zeros(cl1_corr[ell].shape) # Effectively remove that ell

        return invcorr

    def Cls_corr_norm(self, cl1):
        res_norm=np.zeros(cl1.shape)
        res_corr=np.zeros(cl1.shape)
        for ell in range(cl1.shape[0]):
            diagcl= np.diag(cl1[ell])
            #if np.any(diagcl<0):
            #    warnings.warn(f'Negative diagonal element in Cls_corr_norm for ell band {ell+self.params.lb_idx_min}, entry {np.where(diagcl<0)}',category=UserWarning, stacklevel=2)
            sqd = np.sqrt(diagcl)
            res_norm[ell] = np.outer(sqd,sqd)
            res_corr[ell] = cl1[ell]/res_norm[ell]

        return res_corr,res_norm

    def compute_inv_corr(self, bgw=None,
                          h0=None, om=None):
        
        """Inverse of Gamma matrix for building inverse of correlation matrix.
        This version is cosmology-dependent.
        
        cls_th_dir : path to cls to be used as theory
                        usually = cls_dir, but we could want to use
                        the mean cls without lensing as theory,
                        or the mean cls in another cosmology, etc,
                        to check for biases"""
        if h0 is None:
            h0 = self.params.H0
        if om is None:
            om = self.params.OM
        if bgw is None:
            bgw = self.params.bgw

        # Creating the theoretical Gl, Norm (from Gl) and Inv_Corr (from Gl) matrices:
        Gl_gg_th = np.load(f'{self.params.cls_th_dir}/Cl_gg_mean.npy')
        Gl_gb_th = bgw*self.theory.Gl_gb_fid 
        Gl_bb_th = np.load(f'{self.params.cls_th_dir}/Cl_bb_mean.npy')
        
        # IM:  will now REPLACE the auto-correlations by NOISE
        #Gl_gg_th=np.load(f'{cls_th_dir}_rand/Cl_gg_mean.npy')
        #Gl_bb_th=np.load(f'{cls_th_dir}_rand/Cl_gg_mean.npy')

        # Binning in the "likelihood binning"
        # Here Gl_gb is changed to the new cosmology, but not the auto-cls
        Gl_gb_th_binado=self.theory.new_Cl(Gl_gb_th, h0, om)
        Gl_gg_th_binado=self.params.rebinning_Cl(Gl_gg_th,type='zz')
        Gl_bb_th_binado=self.params.rebinning_Cl(Gl_bb_th,type='DfidDfid')

        # Stacking and cutting unwanted bins and multipoles
        Gl_th_full_masked = self.params.stack_Cls(Gl_gb_th_binado,Gl_gg_th_binado,Gl_bb_th_binado)

        # Computing the correlation, inverse correlation and normalization matrices
        
        # Increase lb_idx_min if negative terms appear in auto-correlations at low ells
        dell_min = 0
        for ell in range(Gl_th_full_masked.shape[0]):
            diagcl= np.diag(Gl_th_full_masked[ell])
            if np.any(diagcl<0):

                dell_min = self.params.lb_idx_min+ell+1
                raise ValueError(f"Negative diagonal element in Gl_th_full_masked for ell band {ell}. Increase lb_idx_min to {dell_min}.")

                #warnings.warn(f'Negative diagonal element in Gl_th_full_masked for ell band {ell}. Increasing lb_idx_min to {dell_min}.',category=UserWarning, stacklevel=2)

        # r_cond to invert the full correlation matrix
        rcond=1e-12 #0.01
        fiducial_corr,fiducial_norm=self.Cls_corr_norm(Gl_th_full_masked) #correlation matrix and its associated normalization matrix (1/sqrt(Cl_ii*Cl_jj))
        fiducial_inv_corr=self.inv_corr(fiducial_corr, rcond) # inverse correlation matrix (Gamma^-1)
        
        # return the norm matrix, Gamma^-1, and Gamma
        return fiducial_norm, fiducial_inv_corr, fiducial_corr, Gl_th_full_masked
        

class cl_theory(object):

    def __init__(self, par):

        self.params = par
                        
        # Loading the average Gl_gb
        self.Gl_gb_fid = np.load(f'{self.params.cls_th_dir}/Cl_gb_mean.npy')

        # Loading the theoretical shot-noise (random) components
        try:
            self.Gl_gb_rand_fid = np.load(f'{self.params.cls_th_dir_rand}/Cl_gb_mean.npy')
        except FileNotFoundError:
            warnings.warn(f'Could not find random Cl_gb mean at {self.params.cls_th_dir_rand}/Cl_gb_mean.npy. Setting it to zero.', category=UserWarning, stacklevel=2)
            self.Gl_gb_rand_fid = 0
        
        self.cl_fid = self.Gl_gb_fid-self.Gl_gb_rand_fid
   

    def new_Cl(self, CL,h0,om,w0=-1,wa=0):
        ''' This function creates another C_l(r_i,r_j) by summing many bins from an original C_l(z_i,z_j).
        
            Args:
                CL ('array') : The original 3D C_l(z,z') matrix from where you'll obtain another C_l. 
        
                h0,om,w0,wa ('float') : cosmological parameters
                
                return ('array') : a cosmology-dependent reshaped version of your original
                    C_l(z,z') into C_l(z,D), where D are luminosity distance bins equivalent
                    to (zmin,zmax) in the fiducial model.
                    This is calculated by rebinning galaxy bins and it is the version used in Santiago de Matos et al. 2025.
                    The only adaptation is to use density contrast maps instead of number counts.
   
        '''
        h=h0/100
        omch2=25*om*h**2/30
        ombh2=5*om*h**2/30
        pars = camb.set_params(H0=h0, omch2=omch2, ombh2=ombh2, NonLinear=camb.model.NonLinear_both)
        pars.set_dark_energy(w=w0, cs2=1.0, wa=wa, dark_energy_model='ppf')
        # Careful: Not adapted to w0wa yet!

        results = camb.get_background(pars)
        z_grid = np.linspace(0.0, 3*self.params.zmax, 10001)
        z_of_D = CubicSpline(results.luminosity_distance(z_grid), z_grid)
        new_z = np.where(self.params.D_zb > 0, z_of_D(self.params.D_zb), 0)
        ztol = 1e-10*max(abs(new_z[-1]), 1.0)
        gal_idx = np.array([bisect.bisect_right(new_z, z+ztol)-1 for z in self.params.gal_zs2])
         
        #### new cosmology distance bin edges ####
        D_idx = np.array([bisect.bisect_right(self.params.D_zb, d)-1 for d in self.params.fid_D])#how many D bins fit inside the fiducial cosmology 'bh_bins' D bins
        if np.max(D_idx)==(len(self.params.zb)-1):
            D_idx=D_idx[D_idx<(len(self.params.zb)-1)]
            D_idx=np.append(D_idx,[(len(self.params.zb)-1)])
        else:
            D_idx=D_idx[D_idx<(len(self.params.zb)-1)]
            
        if np.max(gal_idx)==(len(self.params.zb)-1):
            gal_idx=gal_idx[gal_idx<(len(self.params.zb)-1)]
            gal_idx=np.append(gal_idx,[(len(self.params.zb)-1)])
        else:
            gal_idx=gal_idx[gal_idx<(len(self.params.zb)-1)]
            
        matrix_bin1 = np.zeros(shape=(CL.shape[0],CL.shape[1], self.params.bh_bins), dtype = 'float32')  # matrix after fiducial D binning
        matrix_bin2 = np.zeros(shape=(CL.shape[0],self.params.gal_bins, self.params.bh_bins), dtype = 'float32') # matrix after z binning
        sub = 0
        #ng_new = np.zeros(self.params.gal_bins)
        
        # Rebinning BBH distance bins
        for i in range(1,len(D_idx)-1):
            #exceed = ((fid_D[i]-new_D[D_idx[i]] )/(new_D[D_idx[i]+1]-new_D[D_idx[i]])) * CL[:,:,D_idx[i]]
            exceed = ((self.params.fid_D[i]-self.params.D_zb[D_idx[i]] )/(self.params.D_zb[D_idx[i]+1]-self.params.D_zb[D_idx[i]])) * CL[:,:,D_idx[i]]
            matrix_bin1[:,:,i-1] = np.sum(CL[:,:,D_idx[i-1]:D_idx[i]], axis=2) + exceed - sub
            sub = exceed
        matrix_bin1[:,:,i] = np.sum(CL[:,:,D_idx[i]:D_idx[i+1]], axis=2)
        sub = 0
        s = 0
        
        # Rebinning galaxy bins
        # The following line was commented out because we use now galaxy density contrast maps
        # It was needed in Santiago de Matos et al. when using number counts, as well as other lines with 'ng' in what follows
        #matrix_bin1 = matrix_bin1*np.broadcast_to(self.params.ng.reshape(1, len(self.params.ng), 1), (matrix_bin1.shape[0], len(self.params.ng), matrix_bin1.shape[2]))
        for i in range(1,len(gal_idx)-1):
            exceed = ((-new_z[gal_idx[i]]+self.params.gal_zs2[i])/(new_z[gal_idx[i]+1]-new_z[gal_idx[i]])) * matrix_bin1[:,gal_idx[i],:]
            matrix_bin2[:,i-1,:] = np.sum(matrix_bin1[:,gal_idx[i-1]:gal_idx[i],:],axis=1) + exceed - sub
            sub = exceed
            ex = ((-new_z[gal_idx[i]]+self.params.gal_zs2[i])/(new_z[gal_idx[i]+1]-new_z[gal_idx[i]])) #* self.params.ng[gal_idx[i]]
            #ng_new[i-1] = np.sum(self.params.ng[gal_idx[i-1]:gal_idx[i]]) + ex - s
            s = ex
            
        matrix_bin2[:,i,:] = np.sum(matrix_bin1[:,gal_idx[i]:gal_idx[i+1],:],axis=1)
        #ng_new[i] = np.sum(self.params.ng[gal_idx[i]:gal_idx[i+1]])

        # Insert Ng in the denominator of Cl; then rebinn Ng and multiply the final matrix by it
        #ng_newvec = np.broadcast_to(ng_new.reshape(1, len(ng_new), 1), (matrix_bin2.shape[0], len(ng_new), matrix_bin2.shape[2]))
    
        return np.real(matrix_bin2) #np.where(ng_newvec >0, np.real(matrix_bin2)/ng_newvec, np.real(matrix_bin2))       


class cl_data(object):

    def __init__(self, par):

        self.params = par
    
    def data_cls(self, data_type='avg'):
        """Compute the matrix with the Cls = Gls-Gls_rand containing
        both auto and cross spectra to be used as data (in z vs Dl space).
        The auto spectra will later be masked.
            cls_dir : path to cls of data
            avg : if True, use the average Cls in cls_dir as data;
                    if False, use the Cls of a single simulation
                    as data. Default = True."""

        if self.params.forecast == False:
            data_type = 'data'

        print(f'Using {data_type} as data Cls')


        if data_type=='avg':
            # Loading the Average Gamma_l_bb, Gamma_l_gg, Gamma_l_gb
            Gl_gg_data=np.load(f'{self.params.cls_data_dir}/Cl_gg_mean.npy')
            Gl_gb_data=np.load(f'{self.params.cls_data_dir}/Cl_gb_mean.npy')
            Gl_bb_data=np.load(f'{self.params.cls_data_dir}/Cl_bb_mean.npy')
               
            print('Loaded Gls')

        elif data_type=='single':
            # For using as data the Cls of a single simulation (the first one)
            Gl_gb_data=np.load(f'{self.params.cls_data_dir}/Cls_gb1.npy')
            Gl_gg_data=np.load(f'{self.params.cls_data_dir}/Cls_gg1.npy')
            Gl_bb_data=np.load(f'{self.params.cls_data_dir}/Cls_bb1.npy')
                    
        elif data_type=='data':
            # For using real data
            try:
                Gl_gb_data=np.load(f'{self.params.cls_data_dir}/cl_gb_data.npy')
            except:
                Gl_gb_data=np.load(self.params.cls_data_dir)

            Gl_gg_data=np.zeros((Gl_gb_data.shape[0], Gl_gb_data.shape[1], Gl_gb_data.shape[1]))
            Gl_bb_data=np.zeros((Gl_gb_data.shape[0], Gl_gb_data.shape[2], Gl_gb_data.shape[2]))

        else:
            print("Error: data type not recognized. Use 'avg' or 'single' or 'data'.")
        
        # Loading the shot-noise (random) components
        try:

            Gl_gg_rand=np.load(f'{self.params.cls_th_dir_rand}/Cl_gg_mean.npy')
            Gl_gb_rand=np.load(f'{self.params.cls_th_dir_rand}/Cl_gb_mean.npy')
            Gl_bb_rand=np.load(f'{self.params.cls_th_dir_rand}/Cl_bb_mean.npy')

        except FileNotFoundError:

            warnings.warn(f'Could not find random Cl_gb mean at {self.params.cls_th_dir_rand}/Cl_gb_mean.npy. Setting it to zero.', category=UserWarning, stacklevel=2)
            Gl_gg_rand=np.zeros(np.shape(Gl_gg_data))
            Gl_gb_rand=np.zeros(np.shape(Gl_gb_data))
            Gl_bb_rand=np.zeros(np.shape(Gl_bb_data))
        
        # Computing the Cls = Gls-Gls_rand and binning the Cl matrices
        # with the "likelihood" binning, in the fiducial model.    
        Cl_gg_data_binado=self.params.rebinning_Cl(CL=Gl_gg_data-Gl_gg_rand, type='zz')
        Cl_bb_data_binado=self.params.rebinning_Cl(CL=Gl_bb_data-Gl_bb_rand, type='DfidDfid')
        Cl_gb_data_binado=self.params.rebinning_Cl(CL=Gl_gb_data-Gl_gb_rand, type='zDfid')
        Cl_data_full=self.params.stack_Cls(Cl_gb_data_binado,Cl_gg_data_binado,Cl_bb_data_binado)

        Cl_data = Cl_data_full
            
        # Masking data Cl_bb and Cl_gg components since we don't want to use them as data
        Cl_data[:,:self.params.gal_bins-self.params.z_cut,:self.params.gal_bins-self.params.z_cut]=0
        Cl_data[:,self.params.gal_bins-self.params.z_cut:,self.params.gal_bins-self.params.z_cut:]=0
        
        return Cl_data #[self.params.lb_idx_min:self.params.lb_idx_max]      




