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
print('Plots the covariance matrix')
print('------------------------------------------------')
print('------------------------------------------------')
t0 = time.time()
#import astropy.io.fits as fits
#import healpy
# import all pathes

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

X,Y=n.meshgrid(xgrid_ext,xgrid_ext)
p.figure(0, (6,5))
p.axes([0.17,0.17,0.75,0.75])
p.scatter(X, Y, s=185, c=n.flip(covor[:20,:20]),marker='s',vmin=-0.5, vmax=2.5)
p.colorbar(label='covariance')
p.xscale('log')
p.yscale('log')
p.xlim((xgrid_ext.min()/1.1, xgrid_ext.max()*1.1))
p.ylim((xgrid_ext.min()/1.1, xgrid_ext.max()*1.1))
p.xlabel(r'$R/R_{500c}$')
p.ylabel(r'$R/R_{500c}$')
p.savefig(os.path.join(fig_dir, 'cov_mat_radii.png'))
p.clf()

FS = 11
p.figure(0, (6,5))
p.imshow(n.flip(covor, axis=1),vmin=-0.5, vmax=2.5)
p.text(-0.7 , -1.5,'kT', rotation=90, fontsize=FS)#, color='white')
p.text( 0.5 , -1.5,r'$M_{500c}$', rotation=90, fontsize=FS) #, color='white'
p.text( 1.5 , -1.5,r'$z$', rotation=90, fontsize=FS)#, color='white')

for jj in n.arange(len(xgrid_ext))[::3]:
	p.text( 2.5+jj , -1.5, str(n.round(xgrid_ext[jj],3)), fontsize=FS, rotation=90)

for jj in n.arange(len(xgrid_ext))[::3]:
	p.text( -3.5, 19.5 -jj , str(n.round(xgrid_ext[jj],3)), fontsize=FS)

p.text( -3, 22.5 ,'kT', fontsize=FS)#, color='white')
p.text( -3, 21.5 ,r'$M_{500c}$', fontsize=FS) #, color='white'
p.text( -3, 20.5 ,r'$z$', fontsize=FS)#, color='white')

p.text(-0.7 , 24,'kT', rotation=90, fontsize=FS)#, color='white')
p.text( 0.5 , 25,r'$M_{500c}$', rotation=90, fontsize=FS) #, color='white'
p.text( 1.5 , 24,r'$z$', rotation=90, fontsize=FS)#, color='white')
p.text( 11  , 24,r'$R/R_{500c}$', fontsize=14)

p.xticks([])
p.yticks([])
p.colorbar(label='covariance')
p.savefig(os.path.join(fig_dir, 'cov_mat.png'))
p.clf()

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
