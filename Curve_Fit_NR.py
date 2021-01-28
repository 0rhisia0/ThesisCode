from scipy.stats import skewnorm, gamma
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.special import erfc

STEP = 0.05
FUNC_EPS = lambda x, p1, p2, p3: p1 * (abs(x + p2)) ** 0.25 + p3
FUNC_OME = lambda x, p1, p2, p3: p1 / (x - p2) + p3
FUNC_ALP = lambda x, p1, p2: -p1 * x ** 0.5 + p2

def skewNormNew(x, xi, omega, alpha, A):
    """
    Skew normal function for fitting and evaluation purposes
    """
    return A * np.exp(-0.5 * (pow((x - xi) / omega, 2.))) / (np.sqrt(2. * np.pi) * omega) * erfc(
        -1. * alpha * (x - xi) / omega / np.sqrt(2.))


""" Deprecated Skew normal function for fitting purposes"""
# def skewNorm(S1, S2, eps_popt, omeg_popt, alp_popt, norm_popt):
#     eps = FUNC_EPS(S1, eps_popt[0], eps_popt[1], eps_popt[2])
#     omeg = FUNC_OME(S1, omeg_popt[0], omeg_popt[1], omeg_popt[2], omeg_popt[3], omeg_popt[4])
#     alp = FUNC_ALP(S1, alp_popt[0], alp_popt[1], alp_popt[2], alp_popt[3])
#     norm = FUNC_NORM(S1, norm_popt[0], norm_popt[1], norm_popt[2], norm_popt[3])
#     return skewnorm.pdf(S2, alp, loc=eps, scale=omeg) * norm


def getData():
    """
    loads Data from text file output by NEST
    """
    f = open("ERDATA.txt", "r")
    lines = f.readlines()
    S1 = np.zeros(len(lines))
    S2 = np.zeros(len(lines))
    for i in range(len(lines)):
        line = lines[i].split()
        S1[i] = float(line[9])
        S2[i] = float(line[13])
    f.close()

    mask = S2 > 0
    S2 = S2[mask]
    S1 = S1[mask]
    mask = S1 > 0
    S2 = S2[mask]
    S1 = S1[mask]

    for i in range(5):
        print(S1[i], S2[i])

    data = np.vstack([S1, S2])
    print(data.shape)
    return data


def fitToS1S2(data):
    """
    fits skew normals to S1, S2 data and
    interpolates parameters as function of S1.
    """
    a = np.arange(1.5, 98.5, 1)  # bin lower walls
    b = np.arange(2.5, 99.5, 1)  # bin upper walls
    S1 = data[0, :]
    S2 = data[1, :]
    xi = np.zeros(len(a))
    ome = np.zeros(len(a))
    alp = np.zeros(len(a))
    norms = np.zeros(len(a))
    S1_param = np.zeros(len(a))
    logS2Min = np.log10(np.min(S2))
    logS2Max = np.log10(np.max(S2))
    for i in tqdm(range(len(b))):
        MIN = a[i]
        MAX = b[i]

        S2a = S2[(MIN < S1) & (S1 < MAX)]  # creates an S1 slice
        hist_data = np.log10(S2a)

        # generate histogram
        bins = np.arange(3, 6, step=STEP)
        y, bins = np.histogram(hist_data, bins)
        norm_fact = np.sum(y)
        bin_centers = (bins + STEP / 2)[:-1]
        popt, pcov = curve_fit(skewNormNew, bin_centers, y, p0=[4, 0.1, 0, norm_fact],
                               bounds=([logS2Min, 0., -3., 0.], [logS2Max, 1., 3., 1e7]))
        xi[i], ome[i], alp[i], norms[i] = popt
        S1_param[i] = (MIN + MAX) / 2


        # S2_fplot = np.arange(3, logS2Max + 1, step=0.05)
        # pdf = skewNormNew(S2_fplot, *popt)
        # fig, ax = plt.subplots()
        # ax.hist(hist_data, bins=20)
        # ax.set_xlim(3, 5.5)
        # ax.set_yscale("log")
        # ax.plot(S2_fplot, pdf)
        # ax.set_ylim(1e0, 1e4)
        # ax.set_xlabel("log10(S2)[phd]")
        # ax.set_ylabel("Counts")
        #
        # ax.set_title(str(MIN) + "<S1[phd]<" + str(MAX))
        # fig.savefig("C:/Users/Ishira/Pictures/LZ/GIF/" + str(i) + "_GREG.png")
        # plt.close(fig)
    a = np.stack((S1_param, xi, ome, alp, norms))
    np.save('Data/NR_Fit/fit_data', a)
    print(a)



def compare(data1, data2):
    """
    compare cross sectional views of input data for fits (checking for right skewness)
    Yet to be addressed with NEST
    """
    a = np.arange(1.5, 90.5, 1)
    b = np.arange(2.5, 91.5, 1)
    S1_G = data1[0,]
    S2_G = np.log10(data1[1,])
    S1_I = data2[0,]
    S2_I = np.log10(data2[1,])
    for i in tqdm(range(len(b))):
        MIN = a[i]
        MAX = b[i]
        mask_G = (MIN < S1_G) & (S1_G < MAX)
        S2c_G = S2_G[mask_G]
        # S1c_G = S1_G[mask_G]
        mask_I = (MIN < S1_I) & (S1_I < MAX)
        S2c_I = S2_I[mask_I]
        # S1c_I = S1_I[mask_I]
        plt.hist(S2c_G, bins=50, alpha=0.5, label="Greg")
        plt.yscale("log")
        plt.xlim(3, 6)
        plt.hist(S2c_I, bins=50, alpha=0.5, label="Ishi")
        plt.xlabel("log10(S2) [phd]")
        plt.legend()
        plt.title(str(MIN) + "<S1[phd]<" + str(MAX))
        plt.savefig("C:/Users/Ishira/Pictures/LZ/GIF/" + str(i) + "_COMP.png")
        plt.close()


def main():
    """
    main loop to load, fit and interpolate slices
    """

    # data = np.load('Data/NR_Fit/GregNR.npy')  # data to be fit loaded from clean numpy file
    # fitToS1S2(data)
    #
    # # Parameter fits
    a = np.load('Data/NR_Fit/fit_data.npy')  # params generated from fit

    # fit epsilon param
    eps_popt, eps_pcov = curve_fit(FUNC_EPS, a[0], a[1])
    plt.scatter(a[0], a[1])
    eps = FUNC_EPS(a[0], eps_popt[0], eps_popt[1], eps_popt[2])
    plt.plot(a[0], eps, color="r")
    plt.ylabel(r'$\xi$')
    plt.xlabel("S1[phd]")
    plt.ylim(2, 5)
    plt.show()

    # fit ome param
    omeg_popt, omeg_pcov = curve_fit(FUNC_OME, a[0], a[2])
    plt.scatter(a[0], a[2])
    omeg = FUNC_OME(a[0], omeg_popt[0], omeg_popt[1], omeg_popt[2])
    plt.plot(a[0], omeg, color="r")
    plt.ylabel(r'$\omega$')
    plt.xlabel("S1[phd]")
    plt.ylim(0, .4)
    plt.show()

    # fit alp param
    alp_popt, alp_pcov = curve_fit(FUNC_ALP, a[0], a[3])
    plt.scatter(a[0], a[3])
    plt.ylim(-3, 3)
    alp = FUNC_ALP(a[0], alp_popt[0], alp_popt[1])
    plt.plot(a[0], alp, color="r")
    plt.ylabel(r'$\alpha$')
    plt.xlabel("S1[phd]")
    plt.show()

    opts = np.asarray([eps_popt, omeg_popt, alp_popt])
    np.save("Data/NR_Fit/NR_popts", opts)  # saves interpolation parameters


if __name__ == "__main__":
    main()
