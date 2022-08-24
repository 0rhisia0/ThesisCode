from scipy.optimize import minimize
import numpy as np
import emcee
import matplotlib.pyplot as plt
import NReval
import EReval
import Likelihood as lk
from numpy.random import randint
from scipy.stats import lognorm, norm, loguniform
from tqdm import tqdm


# def log_like(xsec, mass, s1, s2, pred_ER):
#     print(mass, xsec, pred_ER)
#     # add cross section effects
#     num_events = len(s1)
#     pois_like, pred_NR = lk.pois_like(mass, xsec, num_events, pred_ER)
#     denom = pred_ER + pred_NR
#     return -(pois_like + np.sum(np.log((pred_NR/denom) * NReval.evaluate_prob(s1, s2, mass) + (pred_NR/denom) * EReval.evaluate_prob(s1, s2))))

def log_like(xsec, mass, s1, s2, pred_ER):
    # add cross section effects
    num_events = len(s1)
    pois_like, pred_NR = lk.pois_like(mass, xsec, num_events, pred_ER)
    denom = pred_NR + pred_ER
    return pois_like + np.sum(np.log((1/denom)*(pred_NR * NReval.evaluate_prob(s1, s2, mass) + pred_ER * EReval.evaluate_prob(s1, s2))))

tmass = 200
events = 0
s1N, s2N = NReval.generateNR(tmass, events)
s1E, s2E = EReval.generateER(1183)
s1 = np.concatenate((s1N, s1E))
s2 = np.concatenate((s2N, s2E))

x = np.linspace(-50, -40, 50)
y = np.linspace(1000, 1400, 20)


mass = 200
mass_ratios = []
for mass in tqdm(np.logspace(1, 3, 50)):
    Z = []
    # find Xsec hat and Nuis hat
    for xsec in x:
        for pred_ER_num in y:
            Z.append(log_like(xsec, mass, s1, s2, pred_ER_num))
    MLE = np.argmax(Z)
    MLE = Z[MLE]
    print(MLE)
    ratios = []
    for xsec in x:
        curr_likes = np.zeros(y.shape[0])
        for i in range(y.shape[0]):
            pred_ER_num = y[i]
            curr_likes[i] = log_like(xsec, mass, s1, s2, pred_ER_num)
        ratios.append(np.max(curr_likes)-MLE)
    mass_ratios.append(ratios)

for mass in tqdm(np.logspace(1, 3, 10)):
    ratios = []
    for xsec in x:
        curr_likes = np.zeros(y.shape[0])
        for i in range(y.shape[0]):
            pred_ER_num = y[i]
            curr_likes[i] = log_like(xsec, mass, s1, s2, pred_ER_num)
        ratios.append(np.max(curr_likes))
        

a = np.asarray(mass_ratios)
print(a)
np.save("Data/Execution files/opt_test.npy",a)