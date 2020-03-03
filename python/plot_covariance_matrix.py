"""
plots the covariance matrix 
"""
import sys
import os
import time
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
import matplotlib.pyplot as p
import matplotlib
from scipy.integrate import quad
from scipy.interpolate import interp1d
from astropy_healpix import healpy
import astropy.io.fits as fits
import numpy as n
print('CREATES SIMPUT CLUSTER FILES')
print('------------------------------------------------')
print('------------------------------------------------')
t0 = time.time()
#import astropy.io.fits as fits
#import healpy
# import all pathes
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 14})


fig_dir = os.path.join( '/home/comparat/software/cluster-brightness-profiles', 'figures')


path_2_cbp = os.path.join(os.environ['GIT_CBP'])
N_clu = 1000
Mpc=3.0856776e+24
msun=1.98892e33
if N_clu < 100:
	nsim = N_clu*1000
else:
	nsim = N_clu*100
covor=n.loadtxt(os.path.join(path_2_cbp, 'covmat_xxl_hiflugcs_xcop.txt'))
xgrid_ext=n.loadtxt(os.path.join(path_2_cbp, 'radial_binning.txt'))
mean_log=n.loadtxt(os.path.join(path_2_cbp, 'mean_pars.txt'))
coolfunc=n.loadtxt(os.path.join(path_2_cbp, 'coolfunc.dat'))

allz_i  = covor[:,len(mean_log)-3]
allm5_i = covor[:,len(mean_log)-2]
allkt_i = covor[:,len(mean_log)-1]


p.figure(0, (6,5))
p.imshow(covor)
p.colorbar()
p.savefig(os.path.join(fig_dir, 'cov_mat.png'))
p.clf()
