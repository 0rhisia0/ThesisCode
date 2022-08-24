from scipy.stats import skewnorm, lognorm, anderson_ksamp, ks_2samp
from scipy import stats
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm
from tabulate import tabulate

ER_POPT = np.load("Data/ER_Fit/ER_popts.npy", allow_pickle=True)
SKEW = ER_POPT[0:3]
NORM = ER_POPT[3]


class FuncNormGen(stats.rv_continuous):
    def _pdf(self, x, p1, p2, p3, p4, A):
        return A ** -1 * (p1 / (x - p4) ** 1.5 + p2 / (x ** 3 - p4) + p3)

    def _argcheck(self, *args):
        """Default check for correct values on args and keywords.
        Returns condition array of 1's where arguments are correct and
         0's where they are not.
        """
        cond = 1
        for arg in args:
            cond = np.logical_and(cond, (np.asarray(arg) > -np.inf))
        return cond


def FUNC_EPS(x, p1, p2, p3):
    """
    FUNC_EPS
    interpolation function for epsilon for skewnorm
    inputs: s1, *popts
    """
    return p1 * (abs(x + p2)) ** 0.25 + p3


def FUNC_OME(x, p1, p2, p3, p4):
    """
    FUNC_OME
    interpolation function for omega for skewnorm
    inputs: s1, *popts
    """
    return p1 / (x - p4) - p2 / (x - p4) ** 2 + p3 * (x - p4)


def FUNC_ALP(x, p1, p2, p3, p4, p5):
    """
    FUNC_ALP
    interpolation function for alpha for skewnorm
    inputs: s1, *popts
    """
    return p1 / (x - p4) - p2 / (x - p4) ** 2 + p3 * (x - p4) + p5


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


def generateER(num):
    """
    generateNR
    Generates S1 and S2 events
    inputs: number of events
    """
    FUNC_NORM = FuncNormGen(name="FUNC_NORM", a=1.5, b=100, shapes="p1, p2, p3, p4, A")
    s1_dist = FUNC_NORM.rvs(*NORM, size=num)
    s2_dist = skewNormGen(s1_dist, *SKEW)
    return s1_dist, s2_dist


def evaluate_prob(s1, s2):
    """
    evaluate_prob
    evaluates the log probability of a given event given on the ER band
    """
    FUNC_NORM = FuncNormGen(name="FUNC_NORM", a=1.5, b=100, shapes="p1, p2, p3, p4, A")
    prob = FUNC_NORM.pdf(s1, *NORM)
    prob2 = skewnorm_eval(s1, s2, *SKEW)
    # print(prob, prob2)
    return prob * prob2


def evaluate_fit(data1, data2):
    # slice the data then evaluate fits for each one.
    S1 = data1[0]
    S2 = np.log10(data1[1])
    S1_2 = data2[0]
    S2_2 = data2[1]
    a = np.arange(0.5, 70.5, 1)
    b = a + 2
    table = []
    for i in range(len(b)):
        MIN = a[i]
        MAX = b[i]
        s1 = MIN + .5
        S2a = S2[(MIN < S1) & (S1 < MAX)]
        S2b = S2_2[(MIN < S1_2) & (S1_2 < MAX)]
        if len(S2a) != 0 and len(S2b)!=0:
            stat = anderson_ksamp([S2a, S2b])
            print(stat)
            # print(stat)
            table.append([s1, stat[0], stat[2]])
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
    return table


def main():
    # dx = 0.5
    # dy = 0.005
    # s1, s2 = np.mgrid[slice(1, 100 + dx, dx), slice(1, 5 + dy, dy)]
    # z = evaluate_prob(s1, s2)
    # fig, ax = plt.subplots()
    # CS = ax.contour(s1, s2, z)
    # ax.clabel(CS, inline=1, fontsize=10)
    # ax.set_title('Simplest default with labels')
    # plt.show()
    ER_actual = np.load("Data/ER_Fit/ER_data_np.npy")
    ER_actual = ER_actual.astype(np.longdouble)
    ER_actual = ER_actual[:, np.random.choice(ER_actual.shape[1], 100, replace=True)]
    mask = ER_actual[0] < 100
    ER_actual = ER_actual[:, mask]
    ER_gen = generateER(20000)
    table = evaluate_fit(ER_actual, ER_gen)
    headers = ["S1 bin [phd]", "k-samp Anderson Statistic", "p-value"]
    print(tabulate(table, headers, tablefmt="latex_longtable"))

    # prob = evaluate_prob(ER_actual[0][:], np.log10(ER_actual[1][:]))
    # print(np.log(np.sum(prob)))

if __name__ == "__main__":
    main()
