import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

samples = np.load("Data/Execution files/samples.npy")
print(samples)

logbins = np.logspace(0
                      , 3, 50)
plt.hist(samples, bins=50)
# plt.yscale("log")
plt.xlim(1, 1000)
# plt.xscale("log")
plt.xlabel(r"Mass $[GeV]$")
plt.ylabel(r"Counts")
plt.axvline(970, color="red")
plt.show()
