"""
plots the covariance matrix 
"""
import sys
import os
import time
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 14})
import matplotlib.pyplot as p

from scipy.integrate import quad
from scipy.interpolate import interp1d
from astropy_healpix import healpy
import astropy.io.fits as fits
import numpy as n
from scipy.stats import scoreatpercentile
print('Plots the covariance matrix')
print('------------------------------------------------')
print('------------------------------------------------')
t0 = time.time()
#import astropy.io.fits as fits
#import healpy
# import all pathes

fig_dir = os.path.join( '/home/comparat/software/cluster-brightness-profiles', 'figures')

pressure_x, pressure_sig_low, pressure_sig_high = n.loadtxt(os.path.join(fig_dir, 'ghirardini_2018_Fig7_pressure.txt'), unpack = True)
entropy_x, entropy_sig_low, entropy_sig_high = n.loadtxt(os.path.join(fig_dir, 'ghirardini_2018_Fig7_entropy.txt'), unpack = True)
density_x, density_sig_low, density_sig_high = n.loadtxt(os.path.join(fig_dir, 'ghirardini_2018_Fig7_density.txt'), unpack = True)

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
nsim = 1000

def calc_lx(prof,kt,m5,z):
	"""
	Compute the X-ray luminosity in the profile
	to be extended to 3x r500c
	
	Compute r_{500c} :
		r_{500c} = \left(\frac{3 M_{500c}}{ 4. \pi 500 \rho_c(z)  }\right)^{1/3} [ cm ].
	
	profile_emission = profile x rescale_factor
	rescale_factor = \sqrt(kT/10.0) E^3(z)
	CF(kT) = cooling function, show the curve
	L_X(r) = \Sigma_{<r}( profile_emission r_{500c}^2 2 \pi x CF(kT) Mpc=3.0856776e+24 dx )
	L_{500c} = L_X(1)
	
	"""
	ez2 = cosmo.efunc(z)**2
	rhoc = cosmo.critical_density(z).value
	r500 = n.power(m5*msun/4.*3./n.pi/500./rhoc,1./3.)
	resfact = n.sqrt(kt/10.0)*n.power(ez2,3./2.)
	prof_em = prof * resfact # emission integral
	tlambda = n.interp(kt,coolfunc[:,0],coolfunc[:,1]) # cooling function
	dx = n.empty(len(xgrid_ext))
	dx[0]=xgrid_ext[0]
	dx[1:len(xgrid_ext)]=(n.roll(xgrid_ext,-1)-xgrid_ext)[:len(xgrid_ext)-1]
	#print(prof_em*xgrid_ext*r500**2*2.*n.pi*tlambda*Mpc*dx)
	lxcum = n.cumsum(prof_em*xgrid_ext*r500**2*2.*n.pi*tlambda*Mpc*dx) # riemann integral
	lx_500=n.interp(1.,xgrid_ext,lxcum) # evaluated at R500
	return lx_500

profs=n.exp(n.random.multivariate_normal(mean_log,covor,size=nsim))

allz_i  = profs[:,len(mean_log)-3]
allkt_i = profs[:,len(mean_log)-1]
allm5_i = profs[:,len(mean_log)-2]

profiles_i = profs[:,:len(xgrid_ext)]

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
cosmoMD = FlatLambdaCDM(
	H0=67.77 * u.km / u.s / u.Mpc,
	Om0=0.307115)  # , Ob0=0.048206)
h = 0.6777
L_box = 1000.0 / h
cosmo = cosmoMD

allz     = allz_i    
allkt    = allkt_i   
allm5    = allm5_i   
profiles = profiles_i
nsim2 = len(allz)
prf_vals = n.zeros((nsim2,len(xgrid_ext)))

name='all'

p.figure(0, (6,5))
p.axes([0.17,0.17,0.75,0.75])
for jj in n.arange(nsim2):
	prf = profiles[jj]
	ez2 = cosmo.efunc(allz[jj])**2
	resfact = n.sqrt(allkt[jj])*n.power(ez2,3./2.)
	p.plot(xgrid_ext, prf/resfact, color='grey')

#p.colorbar(label='covariance')
p.xscale('log')
p.yscale('log')
p.xlim((xgrid_ext.min()/1.1, xgrid_ext.max()*1.1))
p.ylim((1e-12, 3e-5))
p.xlabel(r'$R/R_{500c}$')
p.ylabel(r'$EM kT^{-1/2} E(z)^{-3}$')
p.title(name)
p.savefig(os.path.join(fig_dir, 'profiles_'+name+'.png'))
p.clf()

p.figure(0, (6,5))
p.axes([0.17,0.17,0.75,0.75])
for jj in n.arange(nsim2):
	prf = profiles[jj]
	ez2 = cosmo.efunc(allz[jj])**2
	resfact = n.sqrt(allkt[jj])*n.power(ez2,3./2.)
	prf_vals[jj] = prf/resfact

for pprr in prf_vals:
	p.plot(xgrid_ext, n.log10( pprr/n.mean(prf_vals,axis=0)), color='grey', alpha=0.2)

#p.colorbar(label='covariance')
p.xscale('log')
#p.yscale('log')
p.xlim((xgrid_ext.min()/1.1, xgrid_ext.max()*1.1))
p.ylim((-2, 2))
p.xlabel(r'$R/R_{500c}$')
p.ylabel(r'$\log_{10}(EM kT^{-1/2} E(z)^{-3}$/(mean profile))')
p.title(name)
p.savefig(os.path.join(fig_dir, 'profiles_div_mean_'+name+'.png'))
p.clf()


normed_profile = n.log10(prf_vals/n.mean(prf_vals,axis=0))
pcs = n.array([5, 32, 50, 68, 95])
pc_vals = scoreatpercentile(normed_profile**2, pcs , axis = 0 )**0.5

p.figure(0, (6,5))
p.axes([0.17,0.17,0.75,0.75])
#p.fill_between(xgrid_ext, y1 = pc_vals[0], y2 = pc_vals[-1], color='blue', alpha=0.2, label=r'Prf 2$\sigma$')
p.fill_between(xgrid_ext, y1 = pc_vals[1], y2 = pc_vals[-2], color='blue', alpha=0.3, label=r'Prf 1$\sigma$')
p.plot(xgrid_ext, pc_vals[2], color='blue', ls='dashed')
# Ghirardini
p.fill_between(pressure_x, y1 = pressure_sig_low, y2 = pressure_sig_high, color='g', alpha=0.4, label='Gh20 P')
p.fill_between(entropy_x, y1 = entropy_sig_low, y2 = entropy_sig_high, color='k', alpha=0.4, label='Gh20 E')
p.fill_between(density_x, y1 = density_sig_low, y2 = density_sig_high, color='r', alpha=0.4, label='Gh20 D')
p.xscale('log')
#p.yscale('log')
p.xlim((xgrid_ext.min()/1.1, xgrid_ext.max()*1.1))
p.legend(loc=0)
p.grid()
#p.ylim((1e-12, 3e-5))
p.xlabel(r'$R/R_{500c}$')
p.ylabel('intrinsic scatter')#r'$\sigma(\log_{10}(EM kT^{-1/2} E(z)^{-3}$/(mean profile)))')
p.title(name)
p.savefig(os.path.join(fig_dir, 'profiles_std_'+name+'.png'))
p.clf()







sys.exit()

m_low = (n.log10(allm5_i)>=13.5) & (n.log10(allm5_i)<14.0)
m_mid = (n.log10(allm5_i)>=14.0) & (n.log10(allm5_i)<14.5)
m_hig = (n.log10(allm5_i)>=14.5) 

t_low = (allkt_i < 1) 
t_mid = (allkt_i >= 1) & (allkt_i < 2)
t_hig = (allkt_i >= 2) 

z_low = (allz_i>=0)   & (allz_i<0.3)
z_mid = (allz_i>=0.3) & (allz_i<0.7)
z_hig = (allz_i>=0.7) & (allz_i<1.2)

selections = n.array([
	m_low & t_low & z_low ,
	m_low & t_low & z_mid ,
	m_low & t_low & z_hig ,
	m_low & t_mid & z_low ,
	m_low & t_mid & z_mid ,
	m_low & t_mid & z_hig ,
	m_low & t_hig & z_low ,
	m_low & t_hig & z_mid ,
	m_low & t_hig & z_hig ,
	#                     ,
	m_mid & t_low & z_low ,
	m_mid & t_low & z_mid ,
	m_mid & t_low & z_hig ,
	m_mid & t_mid & z_low ,
	m_mid & t_mid & z_mid ,
	m_mid & t_mid & z_hig ,
	m_mid & t_hig & z_low ,
	m_mid & t_hig & z_mid ,
	m_mid & t_hig & z_hig ,
	#                     ,
	m_hig & t_low & z_low ,
	m_hig & t_low & z_mid ,
	m_hig & t_low & z_hig ,
	m_hig & t_mid & z_low ,
	m_hig & t_mid & z_mid ,
	m_hig & t_mid & z_hig ,
	m_hig & t_hig & z_low ,
	m_hig & t_hig & z_mid ,
	m_hig & t_hig & z_hig 
	])

"kT<1"
"1<kT<2"
"2<kT  "

selection_names = n.array([
	"13.5<M<14 & kT<1 & z<0.3 " ,
	"13.5<M<14 & kT<1 & 0.3<z<0.7" ,
	"13.5<M<14 & kT<1 & 0.7<z<1.2" ,
	"13.5<M<14 & 1<kT<2 & z<0.3 " ,
	"13.5<M<14 & 1<kT<2 & 0.3<z<0.7" ,
	"13.5<M<14 & 1<kT<2 & 0.7<z<1.2" ,
	"13.5<M<14 & 2<kT & z<0.3 " ,
	"13.5<M<14 & 2<kT & 0.3<z<0.7" ,
	"13.5<M<14 & 2<kT & 0.7<z<1.2" ,
	#"#   _&_     _&_      " ,
	"14<M<14.5 & kT<1 & z<0.3 " ,
	"14<M<14.5 & kT<1 & 0.3<z<0.7" ,
	"14<M<14.5 & kT<1 & 0.7<z<1.2" ,
	"14<M<14.5 & 1<kT<2 & z<0.3 " ,
	"14<M<14.5 & 1<kT<2 & 0.3<z<0.7" ,
	"14<M<14.5 & 1<kT<2 & 0.7<z<1.2" ,
	"14<M<14.5 & 2<kT & z<0.3 " ,
	"14<M<14.5 & 2<kT & 0.3<z<0.7" ,
	"14<M<14.5 & 2<kT & 0.7<z<1.2" ,
	#"#   _&_     _&_      " ,
	"14.5<M & kT<1 & z<0.3 " ,
	"14.5<M & kT<1 & 0.3<z<0.7" ,
	"14.5<M & kT<1 & 0.7<z<1.2" ,
	"14.5<M & 1<kT<2 & z<0.3 " ,
	"14.5<M & 1<kT<2 & 0.3<z<0.7" ,
	"14.5<M & 1<kT<2 & 0.7<z<1.2" ,
	"14.5<M & 2<kT & z<0.3 " ,
	"14.5<M & 2<kT & 0.3<z<0.7" ,
	"14.5<M & 2<kT & 0.7<z<1.2" 
	])
for in_zbin, name in zip(selections, selection_names):
	allz     = allz_i    [in_zbin]
	allkt    = allkt_i   [in_zbin]
	allm5    = allm5_i   [in_zbin]
	profiles = profiles_i[in_zbin]
	nsim2 = len(allz)
	prf_vals = n.zeros((nsim2,len(xgrid_ext)))
	if nsim2>0:
		p.figure(0, (6,5))
		p.axes([0.17,0.17,0.75,0.75])
		for jj in n.arange(nsim2):
			prf = profiles[jj]
			ez2 = cosmo.efunc(allz[jj])**2
			resfact = n.sqrt(allkt[jj])*n.power(ez2,3./2.)
			p.plot(xgrid_ext, prf/resfact, color='grey')

		#p.colorbar(label='covariance')
		p.xscale('log')
		p.yscale('log')
		p.xlim((xgrid_ext.min()/1.1, xgrid_ext.max()*1.1))
		p.ylim((1e-12, 3e-5))
		p.xlabel(r'$R/R_{500c}$')
		p.ylabel(r'$EM kT^{-1/2} E(z)^{-3}$')
		p.title(name)
		p.savefig(os.path.join(fig_dir, 'profiles_'+name+'.png'))
		p.clf()

		p.figure(0, (6,5))
		p.axes([0.17,0.17,0.75,0.75])
		for jj in n.arange(nsim2):
			prf = profiles[jj]
			ez2 = cosmo.efunc(allz[jj])**2
			resfact = n.sqrt(allkt[jj])*n.power(ez2,3./2.)
			prf_vals[jj] = prf/resfact
		
		p.plot(xgrid_ext, n.std(prf_vals, axis=0), color='grey')

		#p.colorbar(label='covariance')
		p.xscale('log')
		p.yscale('log')
		p.xlim((xgrid_ext.min()/1.1, xgrid_ext.max()*1.1))
		p.ylim((1e-12, 3e-5))
		p.xlabel(r'$R/R_{500c}$')
		p.ylabel(r'$EM kT^{-1/2} E(z)^{-3}$')
		p.title(name)
		p.savefig(os.path.join(fig_dir, 'profiles_scatter_'+name+'.png'))
		p.clf()
