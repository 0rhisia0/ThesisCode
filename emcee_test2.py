import emcee
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import NReval
import EReval
import Likelihood as lk
from numpy.random import randint
from scipy.stats import lognorm, norm, loguniform


tmass = 50
events = 10

# prior for evaluation. Rejects proposal point if prior is less than 0
def log_prior(theta):
    mass = theta[0]
    xsec = theta[1]
    if 3 < mass <= 1000 and -40 >= xsec >= -50:
        return loguniform.logpdf(mass, 3, 1000)
    return -np.inf

# log_likelihood of current parameters
def log_like(theta, s1, s2, pred_ER):
    # add cross section effects
    mass = theta[0]
    xsec = theta[1]
    num_events = len(s1)
    pois_like, pred_NR = lk.pois_like(mass, xsec, num_events, pred_ER)
    denom = pred_ER + pred_NR
    return pois_like + np.sum(np.log((pred_NR/denom) * NReval.evaluate_prob(s1, s2, mass) + (pred_NR/denom) * EReval.evaluate_prob(s1, s2)))

# log_probability of current parameters given data
def log_probability(theta, data, predER):
    s1=data[0][:]
    s2=data[1][:]
    lp = log_prior(theta)
    if lp == -np.inf:
        return -np.inf
    ll = log_like(theta, s1, s2, predER)
    return lp + ll

#
def proposal(vals, random):
    new_vals = np.empty_like(vals)
    factors = np.full((vals.shape[0]), 0)
    for i in range(vals.shape[0]):
        new_vals[i][0] = lognorm.rvs(0.3, loc=0, scale=vals[i][0])
        new_vals[i][1] = norm.rvs(vals[i][1], 1)
    return new_vals, factors


# data = np.load("Data/NR_Fit/Mass Data/1000G.npy")
# data = data[:, 0:12]
# data = (data[0], np.log10(data[1]))

s1N, s2N = NReval.generateNR(tmass, events)
s1E, s2E = EReval.generateER(1183)
plt.scatter(s1E, s2E, alpha=0.2, color="blue")
plt.scatter(s1N, s2N, alpha=0.2, color="red")
plt.xlabel(r"$S1[phd]$")
plt.ylabel(r"$log_{10}(S2)$")
plt.savefig("C:/Users/Ishira/Pictures/LZ/December/"+str(tmass)+"GeV_signal_"+str(events)+"events")
plt.show()

s1 = np.concatenate((s1N, s1E))
s2 = np.concatenate((s2N, s2E))
data = (s1, s2)

ndim = 2
nwalkers = 5
steps = 50000
pos = [[randint(3, 1000), randint(-49, -36)+0.001] for i in range(nwalkers)]
predER = 1100
cov = np.array([[1000, 0], [0, 1]])
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=[data, predER], moves=emcee.moves.MHMove(proposal, ndim=2))

sampler.run_mcmc(pos, steps, progress=True)
tau = sampler.get_autocorr_time()
print(tau)

flat_samples = sampler.get_chain(discard=0, flat=True)
print(flat_samples.shape)
np.save("Data/Execution files/4", flat_samples)

# mass, xsec, num_events




