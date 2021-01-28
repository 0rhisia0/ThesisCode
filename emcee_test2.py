import emcee
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import NReval
import EReval
import Likelihood as lk
from numpy.random import randint
from scipy.stats import lognorm, norm, loguniform


tmass = 10
events = 1

# prior for evaluation. Rejects proposal point if prior is less than 0
def log_prior(theta):
    mass = theta[0]
    xsec = theta[1]
    if 3 < mass <= 1000 and -35 >= xsec >= -50:
        return 0.0+loguniform.logpdf(mass, 3, 1000)
    return -np.inf

# log_likelihood of current parameters
def log_like(theta, s1, s2, pred_ER):
    # add cross section effects
    mass = theta[0]
    xsec = theta[1]
    num_events = len(s1)
    pois_like, pred_NR = lk.pois_like(mass, xsec, num_events, pred_ER)
    return pois_like + np.sum(np.log(pred_NR * NReval.evaluate_prob(s1, s2, mass) + pred_ER * EReval.evaluate_prob(s1, s2)))

# log_probability of current parameters given data
def log_probability(theta, data, predER):
    s1=data[0][:]
    s2=data[1][:]
    lp = log_prior(theta)
    if lp == -np.inf:
        return -np.inf
    ll = log_like(theta, s1, s2, predER)
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

s1N, s2N = NReval.generateNR(tmass, events)
print(s1N)
s1E, s2E = EReval.generateER(205)
s1 = np.concatenate((s1N, s1E))
s2 = np.concatenate((s2N, s2E))
data = (s1, s2)

plt.scatter(*data, alpha=0.2)
plt.xlabel(r"$S1[phd]$")
plt.ylabel(r"$log_{10}(S2)$")
# plt.savefig("C:/Users/Ishira/Pictures/LZ/December/"+str(tmass)+"GeV_signal_"+str(events)+"events")
plt.show()
#
ndim = 2
nwalkers = 10
steps = 1000
pos = [[randint(3, 1000), randint(-49, -36)+0.001] for i in range(nwalkers)]
predER = 202

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=[data, predER],
                                moves=emcee.moves.WalkMove()) # emcee.moves.MHMove(proposal, ndim=2))
sampler.run_mcmc(pos, steps)

flat_samples = sampler.get_chain(discard=1000, flat=True)
print(flat_samples.shape)
np.save("Data/Execution files/samples1.npy", flat_samples)

# mass, xsec, num_events



