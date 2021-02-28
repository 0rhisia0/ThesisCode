from scipy.optimize import curve_fit
from scipy.stats import lognorm
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

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
    for data in files:
        mass = int(data[:-5])  # pull mass value from string
        vals = np.load("Data/NR_Fit/Mass Data/" + data)  # loads data
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

        # test_plot(xvals, norms, norm_popt, mass)  # optional test plots

        popts.append([mass, *norm_popt])

    popts = np.asarray(popts)
    optimums = []

    for i in range(1, 3):
        plt.scatter(popts[:, 0], popts[:, i])
        opts, cov = curve_fit(FUNC_log, popts[:, 0], popts[:, i])
        xvals = np.arange(3, 1000)

        optimums.append(opts)
        print(FUNC_log(xvals, *opts)[0])
        plt.plot(xvals, FUNC_log(xvals, *opts))
        plt.xlabel("WIMP Mass [GeV]")
        if i == 1:
            plt.ylabel("Shape metric")
        else:
            plt.ylabel("Scale metric")
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
    plt.plot(testx, estim)
    max = xvals[np.argmax(norms)]
    plt.vlines(max, 0, np.max(norms))
    plt.xlabel("S1[phd]")
    plt.ylabel("Counts")
    plt.title(str(mass)+"GeV")
    plt.show()


if __name__ == "__main__":
    main()
