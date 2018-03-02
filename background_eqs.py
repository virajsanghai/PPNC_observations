#  background_PPNC_4.py - Changed units and solve ODE with respect to ln(a)
#  
#
#  Created by Viraj Sanghai on 25/01/2018.
#
import time
import math
import numpy as np
import pylab as p
import scipy.integrate

from scipy.integrate import odeint
#from scipy.integrate import quad
import scipy.optimize
#from numpy.polynomial import polynomial
from numpy import log as log
from numpy import sqrt as sqrt
import pylab as P

start = time.clock()

# Code notations and conventions
# We will use 'a' to represent the scale factor and 'z' to represent the redshift.
# 'k' will denote the spatial curvature. We choose the scale factor today, a_0 to be 1.
# H0 is the Hubble rate today. Omega_M0 is the total energy density of all matter today.
#
# There are 4 PPNC parameters - alpha, alpha_c, gamma, gamma_c.
# Since we have one constraint on the parameters due to energy-momentum conservation
# we only need to specify 3 of them out of which two are given by:
# alpha = alpha_0 a^(-p) ,
# gamma = gamma_0 a^(-q),
# where alpha_0 and gamma_0 are the values of the parameters today and p and q are constants.
# The third parameter is gamma_c which has a contriution from both the matter sector and
# dark energy-like contribution and is represnted by:
# gamma_c = gamma_cm a^(-3) + gamma_cDE,
# where gamma_cm is the matter contribution and  gamma_cDE is the dark enrgy contribution.
# Motivated by Brans-Dicke theory we assume that gamma_cm scales in the same way as gamma so that
# gamma_cm = gamma_cm0 a^(-q) and the dark energy contribution scales as
# gamma_cDE = gamma_cDE0 a^(-r),
# where r is a constant.
# The fourth PPNC parameter is alpha_c, which is fully specified once we have
# alpha, gamma, gamma_c, H0 and Omega_M0
# For LCDM, gamma_cm = 0 and alpha = gamma =1.

# Hence we have a function that requires us to input the standard cosmological parameters
# H0, Omega and k. We also need to input the value of the PPNC parameters and their scaling
# relationship with the scale factor p, q, r.

#Aim: Evaluate growth index in PPNC model

#Define function to input values for parameters in model
def PPNC2(z, H0, Omega_M0, k, alpha_0, p, gamma_0, gamma_cm0, q, gamma_cDE0, r):
    
    #Print parameter values, z is ending redshift
    print("redshift =", z_final)
    print("H0 =", H0)
    print("Omega_M0 =", Omega_M0)
    print("k =", k)
    print("alpha_0 =", alpha_0)
    print("Scaling of alpha, p =", p)
    print("gamma_0 =", gamma_0)
    print("gamma_cm0 =", gamma_cm0)
    print("Scaling of gamma and gamma_cm, q =", q)
    print("gamma_cDE0 =", gamma_cDE0)
    print("Scaling of gamma_DE, r =", r)
    
    #range of redshift values from 0 to z.
    zs = np.linspace(0, z_final, 500)
    
    # The scale factor for different redshifts
    a_vals = 1.0/(1.0+zs)
    
    # Define functions for 3 of the PPNC parameters
    #alpha
    def alphaf(a):
        return alpha_0*(a**(-p))
    
    #gamma
    def gammaf(a):
        return gamma_0*(a**(-q))
    
    #Matter contribution to gamma_c
    def gamma_cmf(a):
        return gamma_cm0*(a**(-q))
    
    #DE contribution to gamma_c
    def gamma_cDEf(a):
        return gamma_cDE0*(a**(-r))
    
    #gamma_c
    def gamma_cf(a):
        return gamma_cmf(a)*(a**(-3.0)) + gamma_cDEf(a)
    
    #The conformal Hubble rate, con_H, using Eq. (7.1) in notes
    def con_Hf(a):
        return ((H0*H0*Omega_M0*gammaf(a))/(a) - 2.0*gamma_cf(a)*(a*a)*(c*c)/3.0 - k*(c*c))**(0.5)
    
    #Conformal time derivative of the conformal Hubble rate, using Eq. (7.12) in notes
    def con_H_primef(a):
        #For this we need the 4th PPNC parameter alpha_c
        #For the 4th PPNC parameter we need the time derivatives gamma and gamma_c
        #Conformal Time derivative of gamma
        gamma_prime = -q*gammaf(a)*con_Hf(a)
        
        #Conformal Time derivative of gamma_prime
        gamma_c_prime = -(3.0 + q)*gamma_cmf(a)*con_Hf(a) -r*gamma_cDEf(a)*con_Hf(a)
        
        #alpha_c can be computed using the constraint equation, Eq. (7.11) in the notes
        alpha_c = (3.0/2.0)*(H0*H0)*Omega_M0*(a**(-3.0))*(alphaf(a)-gammaf(a) + (gamma_prime/con_Hf(a)))  -2.0*gamma_cf(a) - (gamma_c_prime/con_Hf(a))
        
        return -(H0*H0*Omega_M0*alphaf(a))/(2.0*a) + alpha_c*(a*a)*(c*c)/3.0
    
    #Calculate log of a values to integrate with respect to log a
    la_vals = np.log(a_vals)
    
    #Define function to evaluate rate of change of growth factor D
    def dD_da(D, la, alphaf, con_Hf, con_H_primef):
        a = (math.exp(la))
        #D[0] = D and D[1] = D_{, a}, return D_{,a} and D_{,aa}, using Eq. (7.9) in notes
        #return np.array([D[1], ((-2.0*(con_Hf(a)*con_Hf(a))*D[1] - con_H_primef(a)*D[1]) + (3.0/2.0)*(H0*H0)*Omega_M0*alphaf(a)*(a**(-2.0))*D[0])/(a*(con_Hf(a)*con_Hf(a)))])
     
        #D[0] = D and D[1] = D_{, ln a}, return D_{,ln a} and D_{,ln a ln a}, using Eq. (7.10) in notes
        return np.array([D[1], ((-(con_Hf(a)*con_Hf(a))*D[1] - con_H_primef(a)*D[1]) + (3.0/2.0)*(H0*H0)*Omega_M0*alphaf(a)*(a**(-1.0))*D[0])/((con_Hf(a)*con_Hf(a)))])
    
    #Set initial conditions for growth factor, D=1, D_{,a}=0.5.
    #D0 = np.array([1.0, 0.5])
    #D0 = np.array([1.0, Omega_M0**0.54545]) # FIXME
    D0 = np.array([1.0, 0.51278103]) #Fixed using CCL growth rate today
    
  
    print('log',la_vals)
    #integrate to obtain growth factor solutions
    Ds = odeint(dD_da, D0, la_vals, args = (alphaf, con_Hf, con_H_primef))
    
    #DS[0] = D, Ds[1] = D_{, ln a}
    #Growth rate, f, using Eq. (7.10) in notes
    f = (Ds[:,1]/Ds[:,0])
    
    #To compute the growth index, compute Omega_m, For this we need
    gamma = gamma_0*(a_vals**(-q))
    
    # Hubble rate
    H = con_Hf(a_vals)/a_vals
    # Omega_M
    Omega_M = Omega_M0*(a_vals**(-3.0))*(H0*H0)/(H*H)
    #Dark energy density parameter today
    Omega_DE0 = 1.0 - Omega_M0
    #Dark energy density parameter
    Omega_DE = Omega_DE0*(H0*H0)/(H*H)
    
    #growth_index, gamma_growth = ln f / ln (gamma*Omega_M)
    gamma_growth = np.log(f)/np.log(gamma*Omega_M)
    print("gamma_growth for different redshifts \n", gamma_growth)
    
    return zs, H, Omega_M, Omega_DE, f, Ds, gamma_growth

# Ignore: Speed of light, c = 3.0e8, H0 = 73.24e3/c
c = 2.99792458e5 #speed of light in km/s
# Using H0 = 73.24 km/s/Mpc,
H0 = 73.24
#H0 in units of G=c=1
#H0 = 0.0002443023433231266

#Dark energy density parameter today
Omega_lam0 = 0.7

#To reproduce LCDM results, evaluate Lambda
# Omega_lam0 = rho_lam / (3 H0^2 / 8 pi G) where rho_lam = (lam c^2)/(8 pi G)
lam = 3.0*Omega_lam0*(H0*H0)/(c*c)
print('lambda',lam)

#Up to redshift
z_final = 5.0

#Energy density of matter today
Omega_M0 =0.3

#Spatial curvature
k = 0.0

# Set parameters to LCDM values for test
alpha_0 =1.0
p = 0.0
gamma_0 = 1.0
gamma_cm0 =0.0
q =0.0
gamma_cDE0 = -lam/2.0
r=0.0

#Call function to evalute growth index for PPNC model
z, H, Omega_M, Omega_DE, f, D, gam = PPNC2(z_final, H0, Omega_M0, k, alpha_0, p, gamma_0, gamma_cm0, q, gamma_cDE0, r)
# ignore: PPNC(2.0, H0, 0.3, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, -lam/2.0, 1.0)


################################################################################
import pyccl as ccl
cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7324, A_s=2.1e-9, n_s=0.96)
ccl_D = ccl.growth_factor(cosmo, 1./(1.+z))
ccl_f = ccl.growth_rate(cosmo, 1./(1.+z))


################################################################################


#Time code in secs
#print("Time taken", time.clock() - start, "secs")

print('D')
#print('H', H)

#P.subplot(111)
P.figure(1)
aa = 1./(1.+z)

#Plot growth factor
P.plot(1./(1. + z), D[:,0], 'r-', lw=1.8)
P.plot(1./(1. + z), ccl_D, 'b--', lw=1.8)
print('ccl_D', ccl_D)

#Plot growth rate
P.plot(1./(1. + z), f, 'm-', lw=1.8)
P.plot(1./(1. + z), ccl_f, 'c--', lw=1.8)
np.set_printoptions(precision = 15)
print('ccl_f', ccl_f)
#print("ccl_f {:.15f}".format(ccl_f[0,0]))

P.xlabel('Scale factor / a')
P.legend(('Growth factor code', 'Growth factor CCL','Growth rate code', 'Growth rate CCL'))

#P.plot(aa[1:], np.diff(np.log(D[:,0])) / np.diff(np.log(aa)), 'y--', lw=2.8)

#P.plot(1./(1. + z), gam, 'g-', lw=1.8)
#See that matter and DE evolves sensibly
P.figure(2)

P.plot(1./(1. + z), Omega_M, 'k-', lw=1.8)
P.plot(1./(1. + z), Omega_DE, 'c--', lw=1.8)

#Matter dark energy equality line at a redshift of about 0.32 for our LCDM model
# z_equality = (Omega_lam / Omega_m)^(1/3) - 1
P.plot(0.75394744112*np.ones(len(Omega_DE)), Omega_DE, 'y--', lw=1.8)
P.xlabel('Scale factor / a')
P.legend(('$\Omega_M$', '$\Omega_{DE}$','Expected Matter - DE equality'))
#P.plot(z, (1.+z)**2. * 0.55)
#P.axhline(0.55, color='k', ls='dotted')
P.figure(3)

gamma_growth_ccl = np.log(ccl_f)/np.log(Omega_M)
P.plot(aa, gam, 'k-', lw=1.8)
P.plot(aa, gamma_growth_ccl, 'r-', lw=1.8)
#P.plot(aa, H, 'r-', lw=1.8)
P.axhline(6.0/11.0, color='k', ls='dotted')
P.xlabel('Scale factor / a')
P.legend(('Growth index code', 'Growth index CCL'))

P.figure(4)

P.plot(Omega_M, gam, 'k-', lw=1.8)
P.plot(Omega_M, gamma_growth_ccl, 'r-', lw=1.8)
P.axhline(6.0/11.0, color='k', ls='dotted')
P.xlabel('Energy Density parameter of matter/ $\Omega_M$')
P.legend(('Growth index code', 'Growth index CCL'))

P.tight_layout()
P.show()

