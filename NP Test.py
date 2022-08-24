import emcee
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import NReval
import EReval
import Likelihood as lk
from scipy.optimize import minimize
from numpy.random import randint
from scipy.stats import lognorm, norm, loguniform
from tqdm import tqdm


tmass = 50
events = 0

# prior for evaluation. Rejects proposal point if prior is less than 0
# log_likelihood of current parameters
def log_like(theta, s1, s2, pred_ER):
    # add cross section effects
    mass = theta[0]
    xsec = theta[1]
    num_events = len(s1)
    pois_like, pred_NR = lk.pois_like(mass, xsec, num_events, pred_ER)
    denom = pred_NR + pred_ER
    A = NReval.evaluate_prob(s1, s2, mass)
    B = EReval.evaluate_prob(s1, s2)
    sum_term = np.sum(np.log(pred_NR/denom * A + pred_ER/denom *B))
    return pois_like + sum_term

def log_like_H0(s1, s2, pred_ER):
    # add cross section effects
    num_events = len(s1)
    pois_like = sp.stats.poisson.logpmf(num_events, pred_ER)
    return pois_like + np.sum(np.log(EReval.evaluate_prob(s1, s2)))



# def proposal(vals, random):
#     new_vals = np.empty_like(vals)
#     factors = np.full((vals.shape[0]), 0)
#     for i in range(vals.shape[0]):
#         new_vals[i][0] = lognorm.rvs(0.3, loc=0, scale=vals[i][0])
#         new_vals[i][1] = norm.rvs(vals[i][1], 1)
#         while log_prior(new_vals[i]) != 0:
#             new_vals[i][0] = lognorm.rvs(0.3, loc=0, scale=vals[i][0])
#             new_vals[i][1] = norm.rvs(vals[i][1], 2)
#     return new_vals, factors


s1N, s2N = NReval.generateNR(tmass, events)
s1E, s2E = EReval.generateER(1100)
s1 = np.concatenate((s1N, s1E))
s2 = np.concatenate((s2N, s2E))
pred_ER = 1100
#
x = np.logspace(0.5, 3, 40)
y = np.linspace(-50, -40, 40)



Z = np.load("C:/Users/Ishira/Documents/DarkModelFinal/venv/Code/Data/testNP2.npy")
# Z = np.exp(Z)
# Z[mask] = 0
print(np.min(Z), np.max(Z))
fig, ax = plt.subplots()
# print(Z[39, 49], Z[0,0])
cf = ax.contourf(x, y, Z.T)
ax.clabel(cf, inline=1, fontsize=10)
ax.set_title('Simplest default with labels')
# plt.scatter(X, Y, marker=".")
# plt.imshow(interpolation='bilinear')
# im = plt.pcolor(X, Y, Z, cmap='gray')
# plt.xscale("log")
# print(Z)
ax.set_xscale("log")
plt.show()
