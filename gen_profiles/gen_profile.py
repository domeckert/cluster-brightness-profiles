import numpy as np


Mpc=3.0856776e+24
msun=1.98892e33

covor=np.loadtxt('covmat_xxl_hiflugcs_xcop.txt')
xgrid_ext=np.loadtxt('radial_binning.txt')
mean_log=np.loadtxt('mean_pars.txt')
coolfunc=np.loadtxt('coolfunc.dat')

def calc_lx(prof,kt,m5,z):
    ez2=cosmo.efunc(z)**2
    rhoc = cosmo.critical_density(z).value
    r500 = np.power(m5*msun/4.*3./np.pi/500./rhoc,1./3.)
    resfact=np.sqrt(kt/10.0)*np.power(ez2,3./2.)
    prof_em=prof*resfact # emission integral
    tlambda=np.interp(kt,coolfunc[:,0],coolfunc[:,1]) # cooling function
    dx=np.empty(len(xgrid_ext))
    dx[0]=xgrid_ext[0]
    dx[1:len(xgrid_ext)]=(np.roll(xgrid_ext,-1)-xgrid_ext)[:len(xgrid_ext)-1]
    lxcum=np.cumsum(prof_em*xgrid_ext*r500**2*2.*np.pi*tlambda*Mpc*dx) # riemann integral
    lx_500=np.interp(1.,xgrid_ext,lxcum) # evaluated at R500
    return lx_500

profs=np.exp(np.random.multivariate_normal(mean_log,covor,size=nsim))

allz=profs[:,len(mean_log)-3]
allkt=profs[:,len(mean_log)-1]
allm5=profs[:,len(mean_log)-2]

profiles=profs[:,:len(mean_log)]


alllx=np.empty(nsim)
for i in range(nsim):
    tprof=profiles[i]
    alllx[i]=calc_lx(tprof,allkt[i],allm5[i],allz[i])



