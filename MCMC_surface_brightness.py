# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Model fitting with MCMC in Python: #
# 
# This notebook illustrates how to use the pymc module to fit a two-component model  
# to observations of the radial surface brightness profile of M31, as originally  
# measured by [Kent et al., 1987](http://adsabs.harvard.edu/abs/1987AJ.....94..306K).
# 
# Running this code requires the following modules and files to be installed and present:
# 
# - [PyLab](http://www.scipy.org/PyLab), Scipy/Numpy etc
# - [pymc](http://pypi.python.org/pypi/pymc/)
# - SB_model.py
# - m31_SB_profile.txt

# <markdowncell>

# ### Imports ###
# Import pylab using the IPy Notebook magic inline function (%inline), and import our model  
# (defined in the file 'SB_model.py') as well as the pymc MCMC fitter and plotting function.  

from pylab import *
import SB_model
from pymc import MCMC, Matplot


# ### Create the MCMC fitter ###
# Here we construct the MCMC object out of our model.  Remember that the model  
# is just a set of interconnected random variables --- the hard work is
# all handled by this MCMC object.

M = MCMC(SB_model)


# ### Accessing variables ###
# Now that we have our MCMC object, which we called 'M', we can access the model's  
# variables and their values.  For non-observed variables (like SB), a call to .value  
# returns the variable's current value, while the values of observed variables don't change,  
# of course.  Here we plot our observations in red, and a realization of our model generated  
# from our priors on the parameters --- i.e. our model before we begin fitting it.

plt.plot(M.r.value, M.SB.value, c='gray', label='model')
plt.scatter(M.r.value, M.mags.value, c='r', label='observations')

plt.gca().invert_yaxis()
plt.xlabel('radius (")')
plt.ylabel('Surface Brightness (mag)')
plt.legend(loc='best')
plt.show()


# ### Fitting the model ###
# To do this, we sample from the posterior distribution built up by the Markov Chain.  
# The first keyword, *iter*, is the number of samples to draw from our posterior.  
# This is a tradeoff between quality and execution time, since more samples is pretty  
# much always better.
# 
# Of course, we only want to sample after the Markov Chain has found the equilibrium  
# distribution, i.e. after our sampling has converged to the true posterior. That's  
# where the *burn* keyword comes in: the first *burn* samples will be thrown away.  
# We want to ensure that *burn* is large enough so that our Markov Chain converges  
# to the true distribution after that many samples.
# 
# MCMC sampling is oftentimes inherently auto-correlated (the current value of  
# sampled parameters influences the next sample), so *thinning* can be helpful.  
# A sample is collected only every *thin* steps of the Markov Chain.  This gets  
# our sampling closer to the independently-distributed ideal.

# reasonable values for this model
# this will take a few seconds to run, so be patient!
M.sample(iter=5e4, burn=1000, thin=100)


# ### Visualizing the Posterior ###
# The pymc Matplot module is a great way to get a quick snapshot of  
# the posterior. For each parameter, it produces three plots:  
# 
# - A trace, which shows the value of the parameter over time as the Markov Chain walks around.  
#   This should be very noisy --- if the trace has a lot of clear structure, you probably need  
#   to increase the number of samples you *burn*.
# - An autocorrelation plot. High autocorrelation values indicate that you probably should  
#   increase the *thinning* parameter, and sample the space more randomly.
# - A histogram, which will approximate the posterior PDF of that parameter.  
#   This plot is the holy grail of this endeavor --- it shows you the mean (or best-fit)  
#   value for each parameter, as well as the 95% confidence interval.


# plot function takes the model (or a single parameter) as an argument:
Matplot.plot(M)
plt.show()


# ### Making inferences about model parameters ###
# The *stats()* function provides an interface to the statistics of our posterior,  
# in the form of a dictionary.  For example, let's find the predicted ratio between  
# effective sizes of the disk and the bulge, and let's also explore how confidently  
# we can determine the effective surface brightness of the disk.

print 'R_effective (bulge) / R_effective (disk) =', \
       M.stats()['r_e_B']['mean'] / M.stats()['r_e_D']['mean']
print 'Effective surface brightness of the bulge: \n', \
       '    Best-fit value:', M.stats()['M_e_B']['mean'], \
       '\n    95% Confidence interval:', M.stats()['M_e_B']['quantiles'][2.5], \
        'to', M.stats()['M_e_B']['quantiles'][97.5]


# ### Visualizing specific realizations of our model ###
# The *trace()* method presents the values of a variable for all of the saved  
# Markov Chain steps. Let's plot up several of these traces, and see how  
# the model changes with different parameter values.

for i in range(50):
    plt.plot(M.r.value, M.trace('SB')[i], c='gray', alpha=.25)

plt.scatter(M.r.value, M.mags.value, c='r') 
plt.gca().invert_yaxis()
plt.xlabel('radius (")')
plt.ylabel('Surface Brightness (mag)')
plt.show()


# ### Visualizing the best-fit model and its components ###
# Now that we have our best-fit, let's see how it decomposes into the components,  
# the bulge and the disk. In other words, we can finally take our results and  
# explore what they imply about the physical system we observed!

# our best-fit parameters
r_e_B = M.stats()['r_e_B']['mean']
r_e_D = M.stats()['r_e_D']['mean']
M_e_B = M.stats()['M_e_B']['mean']
M_e_D = M.stats()['M_e_D']['mean']
n     = M.stats()['n']['mean']
r     = M.r.value

from SB_model import sersic
# the sersic profile defined in SB_model.py expects brightnesses
#  in flux units, not magnitudes, so we have to convert M_e_B and
#  M_e_D before feeding them in.
plt.plot(r, M.stats()['SB']['mean'], c='k')
plt.plot(r, -2.5*np.log10(sersic( r, r_e_B, n, 10**(-.4*M_e_B) )), 'g:', label='bulge')
plt.plot(r, -2.5*np.log10(sersic( r, r_e_D, 1., 10**(-.4*M_e_D) )), 'g--', label='disk')
plt.scatter(M.r.value, M.mags.value, c='r', label='obs')

plt.gca().invert_yaxis()
plt.axis( [0, 6000, 26, 14] )
plt.ylabel('Surface Brightness (mag)')
plt.xlabel('Radius (")')
plt.legend(loc='best')
plt.show()
