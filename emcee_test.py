import emcee
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import NReval
import Likelihood as lk
from numpy.random import randint
from scipy.stats import lognorm, norm, loguniform
tmass = 10
events = 10

# prior for evaluation. Rejects proposal point if prior is less than 0
def log_prior(theta):
    mass = theta[0]
    xsec = theta[1]
    if 3 < mass <= 1000 and -35 >= xsec >= -50:
        return 0.0+loguniform.logpdf(mass, 3, 1000)
    return -np.inf

# log_likelihood of current parameters
def log_like(theta, s1, s2):
    # add cross section effects
    mass = theta[0]
    xsec = theta[1]
    num_events = len(s1)
    pois_like = lk.pois_like_S1S2(mass, xsec, num_events)
    return pois_like+np.sum(NReval.evaluate_prob(s1, s2, mass))

# log_probability of current parameters given data
def log_probability(theta, s1, s2):
    lp = log_prior(theta)
    if lp == -np.inf:
        return -np.inf
    ll = log_like(theta, s1, s2)
    return lp + ll


def proposal(vals, random):
    new_vals = np.empty_like(vals)
    factors = np.full((vals.shape[0]), 0)
    for i in range(vals.shape[0]):
        new_vals[i][0] = lognorm.rvs(0.3, loc=0, scale=vals[i][0])
        new_vals[i][1] = norm.rvs(vals[i][1], 1)
        while log_prior(new_vals[i]) != 0:
            new_vals[i][0] = lognorm.rvs(0.3, loc=0, scale=vals[i][0])
            new_vals[i][1] = norm.rvs(vals[i][1], 2)
    return new_vals, factors


# data = np.load("Data/NR_Fit/Mass Data/1000G.npy")
# data = data[:, 0:12]
# data = (data[0], np.log10(data[1]))

data = NReval.generateNR(tmass, events)

plt.scatter(*data, alpha=0.2)
plt.xlabel(r"$S1[phd]$")
plt.ylabel(r"$log_{10}(S2)$")
plt.savefig("C:/Users/Ishira/Pictures/LZ/December/"+str(tmass)+"GeV_signal_"+str(events)+"events")
plt.show()

ndim = 2
nwalkers = 5
steps = 2000
pos = [[randint(3, 1000), randint(-49, -36)+0.001] for i in range(nwalkers)]


sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=data)   # moves=emcee.moves.MHMove(proposal, ndim=2)
sampler.run_mcmc(pos, steps)

flat_samples = sampler.get_chain(discard=100, flat=True)
print(flat_samples.shape)
np.save("Data/Execution files/samples1", flat_samples)

# mass, xsec, num_events



