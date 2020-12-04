import emcee
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import NReval
tmass = 970
events = 10


def log_prior(mass):
    if 3 < mass <= 1000:
        return 0.0
    return -np.inf


def log_like(mass, s1, s2):
    return np.sum(NReval.evaluate_prob(s1, s2, mass))


def log_probability(mass, s1, s2):
    lp = log_prior(mass)
    if lp == -np.inf:
        return -np.inf
    ll = log_like(mass, s1, s2)
    return lp + ll


# data = np.load("Data/NR_Fit/Mass Data/1000G.npy")
# data = data[:, 0:12]
# data = (data[0], np.log10(data[1]))
data = NReval.generateNR(tmass, events)

plt.scatter(*data, alpha=0.2)
plt.xlabel(r"$S1[phd]$")
plt.ylabel(r"$log_{10}(S2)$")
plt.savefig("C:/Users/Ishira/Pictures/LZ/December/"+str(tmass)+"GeV_signal_"+str(events)+"events")
plt.show()

ndim = 1
nwalkers = 5
steps = 1000
pos = [[i] for i in np.random.randint(10, 1000, nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=data)
sampler.run_mcmc(pos, steps)

flat_samples = sampler.get_chain(discard=100, flat=True)
print(flat_samples.shape)
np.save("Data/Execution files/samples", flat_samples)
