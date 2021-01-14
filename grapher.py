import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

samples = np.load("Data/Execution files/samples1.npy")

# bins = [np.logspace(0, 3, 500), np.linspace(-50, -35, 500)]
samples = samples[:, :]
plt.scatter(samples[800:, 0], samples[800:, 1], alpha=0.01)
plt.xlim(1, 1000)
plt.ylim(-49, -43)
plt.xscale("log")
# plt.hexbin(samples[:, 0], samples[:, 1], xscale="log")
plt.xlabel(r"Mass $[GeV]$")
plt.ylabel(r"$log_{10}(XSec)$")
# plt.axvline(970, color="red")
plt.show()
