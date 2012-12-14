"""
A pymc model for Galaxy surface-brightness profiles, including
 observations by Kent et al. (http://adsabs.harvard.edu/abs/1987AJ.....94..306K).
 This file only defines the model, and depends on other codes to 
 fit and analyze it.

A pymc model is a set of connected random variables and their
 probability distributions. A model is defined with two major 
 types of pymc objects:
 Stochastic: any variable with intrinsic randomness
 Deterministic: any variable whose value is completely determined by inputs (no randomness)

Our model below includes several stochastic variables:
 the parameters of our model: r_e_B, r_e_D, M_e_B, M_e_D, and n
 our observed values: r and mags
and one deterministic variable:
 the output of our surface brightness profile: SB

You can find more examples and read more about pymc models at
 http://pymc-devs.github.com/pymc/index.html
"""


#########################################################
# Imports
import pymc
import numpy as np


#########################################################
# Define the model we want to fit to our observations
def sersic( r, r_e, n, I_e ):
    '''
    The Sersic profile. Returns array-like I(r), in whatever
     units I_e has.
    
    r: radius, array-like
    r_e: effective radius, a scale size
    n: sersic index; .5 < n < 8
    I_e: surface brightness at r=r_e
    '''
    # first calculate the b_n value, from
    #  MacArthur, Courteau, & Holtzman (2003):
    b_n = 2.*n - 1./3 + 4./(405*n) + 46./(25515*n**2) + 131./(1148175*n**3) \
              - 2194697./(30690717750*n**4)
    I = I_e * np.exp( -b_n * ((r/r_e)**(1./n) - 1.) )
    return I

def full_profile( r, r_e_B, r_e_D, n, I_e_B, I_e_D ):
    '''
    The mash-up of one Sersic profile with n as a free parameter,
     and one Sersic profile with n=1 (i.e. an exponential disk).
     Good fit for disk galaxies with a bulge and a disk.
     Returns I(r).
    
    r: radius, array-like
    r_e_B: effective bulge radius, a scale size
    r_e_D:  ...      disk  ...
    n: sersic index for bulge; .5 < n < 10
    I_e_B: bulge surface brightness at r=r_e_B
    I_e_D: disk  ...
    '''
    I_bulge = sersic( r, r_e_B, n, I_e_B )
    I_disk  = sersic( r, r_e_D, 1., I_e_D )
    I = I_bulge + I_disk
    return I


#########################################################
# Load our observations and define
#  all of our pymc random variables
dat = np.loadtxt('m31_SB_profile.txt')
measured_radii = dat[:,0] #arcseconds
measured_mags  = dat[:,1] #magnitudes

# Define our priors on parameters
#  For this example, these priors are remarkably uninformative.
r_e_B = pymc.Uniform('r_e_B', lower=0., upper=np.mean(measured_radii), 
                            doc='effective radius of bulge [arcsec]')
r_e_D = pymc.Uniform('r_e_D', lower=np.mean(measured_radii), upper=measured_radii[-1], 
                            doc='effective radius of disk [arcsec]')
M_e_B = pymc.Uniform('M_e_B', lower=np.mean(measured_mags), upper=max(measured_mags),
                            doc='intensity of bulge at bulge effective radius')
M_e_D = pymc.Uniform('M_e_D', lower=np.mean(measured_mags), upper=max(measured_mags),
                            doc='intensity of disk at disk effective radius')
n     = pymc.Uniform('n', lower=.25, upper=10., doc='Sersic index of bulge')

# Define our radii observations as observed random variables.
#  Though no radius measurement errors were given, I assume 
#  that the authors were able to measure spatial scales at about the .5-pixel
#  level, which depends on which telescope was used (see text).
#  I model these as Gaussian errors about the observed value.
err_map = {0:.31/2, 1:.73/2, 2:2.95/2, 3:6.03/2, 4:58./2}
err_radius = np.array( [err_map[ int(val) ] for val in dat[:,3]] )
# r is the random variable representing the radii of our observations.
#  These are observed random variables drawn from Gaussian distributions
#  with mean values of the observed value and variances as defined by the assumed errors.
r = pymc.Normal('r', mu=measured_radii, tau=1./err_radius, value=measured_radii,
                        observed=True, doc='Observed radius')

# SB is our surface brightness profile model, in magnitudes. A function of
#  radius and all of the parameters of our models.
#  This also shows the deterministic decorator in pymc, which is very helpful
#  for deterministic variables (those that are functions of other variables).
@pymc.deterministic(plot=False)
def SB( r=r, r_e_B=r_e_B, r_e_D=r_e_D, n=n, I_e_B=10**(-.4*M_e_B), I_e_D=10**(-.4*M_e_D)):
    '''
    Model of surface brightness in magnitudes.
    '''
    I = full_profile( r, r_e_B, r_e_D, n, I_e_B, I_e_D )
    return -2.5*np.log10(I)

# Define our surface brightness observations as observed random variables.
#  Only minimal error estimates were given in the text.
#  The authors estimated their errors to range between 0.03mag for
#  the bright regions up to 0.1mag for the dim regions
mag_err = np.linspace(0.03, .1, len(measured_mags))
# mags is the random variable representing the brightnesses of our observations.
#  These are observed random variables drawn from Gaussian distributions
#  with mean values defined by our model and variances as defined by the assumed errors.
mags = pymc.Normal('mags', mu=SB, tau=1./mag_err, value=measured_mags,
                        observed=True, doc='Observed surface brightness')

