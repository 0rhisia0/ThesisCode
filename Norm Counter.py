from scipy.optimize import curve_fit
from scipy.stats import lognorm, anderson_ksamp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import NReval
from tabulate import tabulate


plt.style.use("seaborn-talk")

FUNC_NORM = lambda x, p1, p2: lognorm.pdf(x, s=p1, scale=p2, loc=0)
FUNC_log = lambda x, p1, p2, p3: -p2/(x-p1)+p3



def main():
    """
    Code loop for fitting normalization in S1 W.R.T some interpolating
    parameter. For example: WIMP mass for NR or g2gas for ER/NR
    """
    a = np.arange(0.5, 70.5, 1)  # S1 bins lower
    b = a + 1  # S1 bins lower
    files = os.listdir("Data/NR_Fit/Mass Data")  # training Data directory
    files.sort()
    print(files)
    popts = []
    table = []
    for data in files:
        mass = int(data[:-5])  # pull mass value from string
        vals = np.load("Data/NR_Fit/Mass Data/" + data)  # loads data
        print(vals.shape)
        S1 = vals[0]
        S2 = vals[1]
        norms = np.zeros(len(b))  # initializes counts
        for i in tqdm(range(len(b))):
            MIN = a[i]
            MAX = b[i]
            S2a = S2[(MIN < S1) & (S1 < MAX)]  # Counts within bins
            norm_fact = len(S2a)
            norms[i] = norm_fact  # sets i'th count
        total = np.sum(norms)
        norms /= total  # normalizes the counts to create a PDF
        xvals = a + 0.5  # Bin centers
        norm_popt, norm_pcov = curve_fit(FUNC_NORM, xvals,
                                         norms, bounds=([-np.inf, -np.inf], [np.inf, np.inf]),
                                         p0=[10, 0.5])
        # fits FUNC_NORM and returns parameters/covariance of fit
        # testing anderson
        evaluated = NReval.generateNR(mass, 300)[0]
        stat = anderson_ksamp([evaluated, S1[:300]])
        table.append([mass, stat[0], stat[2]])
        test_plot(xvals, norms, norm_popt, mass)  # optional test plots
        popts.append([mass, *norm_popt])
    table = sorted(table, key=lambda x: x[0])
    headers = ["S1 bin [phd]", "k-samp Anderson Statistic", "p-value"]
    print(tabulate(table, headers, tablefmt="latex_longtable"))
    popts = np.asarray(popts)
    optimums = []

    for i in range(1, 3):
        plt.scatter(popts[:, 0], popts[:, i])
        opts, cov = curve_fit(FUNC_log, popts[:, 0], popts[:, i])
        xvals = np.arange(3, 1000)

        optimums.append(opts)
        print(opts)
        plt.plot(xvals, FUNC_log(xvals, *opts), color="black")
        plt.xlabel("WIMP Mass [GeV]", fontsize=18)
        plt.xscale("log")
        if i == 1:
            plt.ylabel(r"Shape metric, $\alpha$", fontsize=18)
            plt.text(20, 0.4, r"$\alpha(m_{\chi}) = \frac{-12.2535678}{m_{\chi}+10.03005277}+ 1.07541932$", fontsize=20)
        else:
            plt.ylabel(r"Scale metric, $\omega$", fontsize=18)
            plt.text(20, 4, r"$\omega(m_{\chi}) = \frac{-562.43584632}{m_{\chi}+26.85837188}+ 18.86300004$",
                     fontsize=20)
        plt.show()

    np.save("Data/NR_Fit/norm_fits", optimums)
    print("Parameters saved")

    # check normalization curves
    while True:
        check = input("Check Normalizations? (yes/no)\n")
        if check == "yes":
            mass = float(input())
            a = FUNC_log(mass, *optimums[0])
            b = FUNC_log(mass, *optimums[1])
            xs = np.arange(0, 70)
            xy = FUNC_NORM(xs, a, b)
            plt.plot(xs, xy)
            plt.show()
        else:
            break

def test_plot(xvals, norms, norm_popt, mass):
    """
    code to test plot the interpolation
    of any fixed parameter S1 normalization curve
    Uncomment on main to run
    """
    plt.scatter(xvals, norms)
    testx = np.arange(0, 70, 0.1)
    estim = FUNC_NORM(testx, *norm_popt)
    plt.plot(testx, estim, color="black")
    max = xvals[np.argmax(norms)]
    plt.xlabel("S1[phd]", fontsize=17)
    plt.ylabel("Counts", fontsize=17)
    plt.title(str(mass)+r" GeV/c^2", fontsize=17)
    plt.show()


if __name__ == "__main__":
    main()
