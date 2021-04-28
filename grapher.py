import scipy as sp
import numpy as np
import matplotlib.pyplot as plt


fig, axes = plt.subplots(3,1, sharex=True)
axes = axes.flatten()
masses = [30, 50, 75, 100]
scatters = [[50, -47.6], [50, -46.6], [50, -45.548]]
# limits = [[-48,-44], [-48,-44], [-48,]]
for i in range(0, 3):
    axis = axes[i]
    samples = np.load("Data/Execution files/" + str(i+1) + ".npy")
    samples = samples[:, :]
    axis.scatter(samples[:, 0], samples[:, 1], alpha=0.1)
    axis.set_xlim(10, 1000)
    axis.set_ylim(-48.5, -44)
    axis.set_xscale("log")
    # plt.scatter()
    # plt.hexbin(samples[:, 0], samples[:, 1], xscale="log")
    axis.set_ylabel(r"$log_{10}(XSec)$")
    axis.scatter(scatters[i][0], scatters[i][1], marker = "x", color="red")


axis.set_xlabel(r"Mass $[GeV]$")

# bins = [np.logspace(0, 3, 500), np.linspace(-50, -35, 500)]



# plt.axvline(970, color="red")
plt.show()

# def find_nearest_idx(array, value):
#     array = array - value
#     idx = (np.abs(array)).argmin()
#     return idx
#
# fig, axes = plt.subplots(10, 1, figsize=(5, 20))
# axes = axes.flat
# masses = np.logspace(1, 3, 10)
# vals = []
# for i in range(10):
#     ax = axes[i]
#     xsec = np.linspace(-50, -40, 200)
#     q = -2*samples[i]
#     vals.append(xsec[find_nearest_idx(q, 2.706)])
#     ax.plot(xsec, q)
#     ax.set_yscale("log")
#     ax.set_title(str(masses[i])[:3] + " GeV")
# plt.show()
# plt.close(fig)
#
# xsec = np.linspace(-50, -40, 200)
# plt.plot(xsec, -2 * samples[3])
# plt.yscale("log")
# plt.ylim(0, 100)
# plt.text(-44, 2.8, r"$\chi^2$ for 90%")
# plt.hlines(2.706, -50, -40)
# plt.title(str(masses[3])[:3] + " GeV")
# plt.show()

# plt.plot(masses, vals)
# plt.xscale("log")
# plt.show()
