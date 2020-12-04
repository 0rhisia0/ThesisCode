from scipy.stats import skewnorm, lognorm, anderson_ksamp, ks_2samp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
SKEW = np.load("Data/NR_Fit/NR_popts.npy", allow_pickle=True)
NORM = np.load("Data/NR_Fit/norm_fits.npy")
plt.style.use("seaborn-talk")


def FUNC_norm(x, p1, p2):
    """
    FUNC_norm
    Evaluates probability of a given S1 using the effective marginalized pdf in S1.
    inputs: s1, shape, scale
    """
    return lognorm.pdf(x, s=p1, scale=p2, loc=0)


def FUNC_norm_gen(p1, p2, num):
    """
    FUNC_norm_gen
    Generated S1 events using the effective marginalized pdf in S1.
    inputs: s1, shape, scale
    """
    return lognorm.rvs(p1, loc=0, scale=p2, size=num)


def FUNC_norm_mass_interp(x, p1, p2, p3):
    """
    FUNC_norm_mass_interp
    interpolation function for both params in the S1 marginalized normalization function (lognorm)
    inputs: mass, popts*
    """
    return -p2/(x-p1)+p3


def FUNC_EPS(x, p1, p2, p3):
    """
    FUNC_EPS
    interpolation function for epsilon for skewnorm
    inputs: s1, popts*
    """
    return p1 * (abs(x + p2)) ** 0.25 + p3


def FUNC_OME(x, p1, p2, p3):
    """
    FUNC_OME
    interpolation function for omega for skewnorm
    inputs: s1, popts*
    """
    return p1/(x-p2) + p3


def FUNC_ALP(x, p1, p2):
    """
    FUNC_OME
    interpolation function for alpha for skewnorm
    inputs: s1, popts*
    """
    return -p1*x**0.5 + p2


def skewNormGen(s1, xi_popts, ome_popts, alp_popts):
    """
    skewNormGen
    Generated S1 events using the effective marginalized pdf in S1.
    inputs: s1, xi optimums, omega optimums, alpha optimums
    """
    xi = FUNC_EPS(s1, *xi_popts)
    omega = FUNC_OME(s1, *ome_popts)
    alpha = FUNC_ALP(s1, *alp_popts)
    return skewnorm.rvs(alpha, loc=xi, scale=omega)


def generateNR(mass, num):
    """
    generateNR
    Generates S1 and S2 events
    inputs: mass of WIMP, number of events
    """
    a = FUNC_norm_mass_interp(mass, *NORM[0])
    b = FUNC_norm_mass_interp(mass, *NORM[1])
    s1_dist = FUNC_norm_gen(a, b, num)
    s2_dist = skewNormGen(s1_dist, *SKEW)
    return s1_dist, s2_dist


def skewnorm_eval(s1, s2, xi_popts, ome_popts, alp_popts):
    """
    skewnorm_eval
    Evaluates probability of S1 and S2 events
    inputs: mass of WIMP, number of events
    """
    xi = FUNC_EPS(s1, *xi_popts)
    omega = FUNC_OME(s1, *ome_popts)
    alpha = FUNC_ALP(s1, *alp_popts)
    return skewnorm.pdf(s2, alpha, loc=xi, scale=omega)


def evaluate_prob(s1, s2, mass):
    """
    evaluate_prob
    evaluates the log probability of a given event given WIMP mass (no poisson factor)
    """
    a = FUNC_norm_mass_interp(mass, *NORM[0])
    b = FUNC_norm_mass_interp(mass, *NORM[1])
    prob = np.log(FUNC_norm(s1, a, b))
    prob2 = np.log(skewnorm_eval(s1, s2, *SKEW))
    if np.isnan(prob+prob2).any():
        print(prob, prob2, "a=", a, "b=", b, "mass=", mass, "Prob")
    return prob+prob2


def sample_cont(mass):
    x = np.linspace(2, 100, 500) # evaluation starts at two to avoid ugly spike, justified by low event rate below this
    y = np.linspace(3, 5, 500)
    xx, yy = np.meshgrid(x, y, sparse=False)
    z = evaluate_prob(xx, yy, mass)
    fig, ax = plt.subplots()
    h = ax.pcolormesh(x, y, z)
    ax.set_title("M_D="+str(mass)[0:4])
    plt.show()
    # fig.savefig("C:/Users/Ishira/Pictures/LZ/NR_morph2/"+str(mass)[0:4]+"log.png")
    # plt.close(fig)

def evaluate_fit(data1, data2):
    # slice the data then evaluate fits for each one.
    S1 = data1[0]
    S2 = np.log10(data1[1])
    S1_2 = data2[0]
    S2_2 = data2[1]
    a = np.arange(0.5, 70.5, 1)
    b = a + 2
    stats = []
    for i in range(len(b)):
        MIN = a[i]
        MAX = b[i]
        s1 = MIN + .5
        S2a = S2[(MIN < S1) & (S1 < MAX)]
        S2b = S2_2[(MIN < S1_2) & (S1_2 < MAX)]
        if len(S2a) != 0 and len(S2b)!=0:
            print("S1="+str(s1))
            stat = anderson_ksamp([S2a, S2b])
            print(stat)
            stats.append(stat[0])
            if len(S2a)>len(S2b):
                S2a = S2a[:len(S2b)]
            else:
                S2b = S2b[:len(S2a)]
            # plt.hist(S2a, bins=100, alpha=0.5)
            # plt.hist(S2b, bins=100, alpha=0.5, color="r")
            # plt.show()
            # if stat[2] < 0.05:
            #     print(len(S2a), len(S2b))
            #     bins = np.linspace(3, 5, 100)
            #     plt.hist(S2a, bins, alpha=0.5, label='x')
            #     plt.hist(S2b, bins, alpha=0.5, label='y')
            #     plt.show()
    average_significance = np.average(stats)
    return average_significance


# for mass in tqdm(np.logspace(1, 3, 10)):
#     sample_cont(mass)
#

# data_true = np.load("100G.npy")
# data_mask = data_true[0] < 100
# data_x = data_true[0][data_mask]
# data_y = data_true[1][data_mask]

# plot comparison between NEST and generated
# fig, ax = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
# ax[0].scatter(data_x[0:1000], np.log10(data_y)[0:1000], alpha=0.1)
# ax[1].scatter(*generateNR(100, 1000), alpha=0.1)
# ax[0].set_xlim(0, 100)
# ax[1].set_xlim(0, 100)
# ax[0].set_title("NEST Data")
# ax[0].set_ylim(3, 5)
# ax[1].set_ylim(3, 5)
# ax[1].set_title("Analytic Data")
# plt.show()

# evaluate fit
# data_true[:10000]
# print(data_true.shape)
# total = 0
# for i in range(10):
#     data_ana = generateNR(1000, 10000)
#     total += evaluate_fit(data_true, data_ana)
# print(total/10)


