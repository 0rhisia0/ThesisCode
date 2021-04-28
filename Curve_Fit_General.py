from scipy.stats import skewnorm, gamma, anderson_ksamp
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.special import erfc
from scipy.stats import lognorm
from scipy.integrate import cumtrapz
import EReval


STEP = 0.05
FUNC_EPS = lambda x, p1, p2, p3: p1 * (abs(x + p2)) ** 0.25 + p3
FUNC_OME = lambda x, p1, p2, p3, p4: p1/(x-p4) - p2/(x-p4)**2 + p3*(x-p4)
FUNC_ALP = lambda x, p1, p2, p3, p4, p5: p1/(x-p4) - p2/(x-p4)**2 + p3*(x-p4)+p5
FUNC_NORM = lambda x, p1, p2, p3, p4: p1/(x-p4)**(1.5) + p2/(x**3-p4) + p3

def skewNormNew(x, xi, omega, alpha, A):
    """
    Skew normal function for fitting and evaluation purposes
    """
    return A * np.exp(-0.5 * (pow((x - xi) / omega, 2.))) / (np.sqrt(2. * np.pi) * omega) * erfc(
        -1. * alpha * (x - xi) / omega / np.sqrt(2.))


def getData(file):
    """
    loads Data from text file output by NEST
    """
    f = open(file, "r")
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
        norms[i] = norm_fact  # retain true value for fitting instead of fitted value
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
    np.save('Data/ER_Fit/ER_fit_data', a)  # saves fit
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
    # data = getData('Data/ER_Fit/ERDATA.txt')
    # np.save('Data/ER_Fit/ER_data_np', data)  # data to be fit loaded from clean numpy file
    data = np.load('Data/ER_Fit/ER_data_np.npy')
    # fitToS1S2(data)

    # Parameter fits
    a = np.load('Data/ER_Fit/ER_fit_data.npy')  # params generated from fit

    # fit epsilon param
    eps_popt, eps_pcov = curve_fit(FUNC_EPS, a[0], a[1])
    plt.scatter(a[0], a[1])
    eps = FUNC_EPS(a[0], eps_popt[0], eps_popt[1], eps_popt[2])
    plt.plot(a[0], eps, color="r")
    print(eps_popt)
    plt.ylabel(r'Location, $\varepsilon$', fontsize=17)
    plt.xlabel("S1[phd]", fontsize=17)
    plt.ylim(2, 5)
    plt.text(5, 3, r"$\varepsilon(S1)\approx 0.4656\cdot (|S1+6.0221|)^{0.25}+3.1364$", fontsize=14)
    plt.show()

    # fit ome param
    omeg_popt, omeg_pcov = curve_fit(FUNC_OME, a[0], a[2])
    plt.scatter(a[0], a[2])
    omeg = FUNC_OME(a[0], omeg_popt[0], omeg_popt[1], omeg_popt[2], omeg_popt[3])
    print(omeg_popt)
    plt.plot(a[0], omeg, color="red")
    plt.ylabel(r'Scale, $\omega$', fontsize=17)
    plt.xlabel("S1[phd]", fontsize=17)
    plt.ylim(0, .4)
    plt.text(0, 0.25, r"$\omega(S1)\approx\frac{114.1}{S1+35.078}-\frac{299.91}{(S1+35.078)^2}+0.00042\cdot(S1+35.078)$", fontsize=12)
    plt.show()

    # fit alp param
    alp_popt, alp_pcov = curve_fit(FUNC_ALP, a[0], a[3])
    print(alp_popt)
    plt.scatter(a[0], a[3])
    plt.ylim(-3, 3)
    alp = FUNC_ALP(a[0], alp_popt[0], alp_popt[1], alp_popt[2], alp_popt[3], alp_popt[4])
    print(alp_popt)
    plt.text(0, -1,
             r"$\alpha(S1)\approx\frac{226.6}{S1+18.91}-\frac{3434.2}{(S1+18.91)^2}+0.0184\cdot(S1+18.91)-2.401$",
             fontsize=11.5)
    plt.plot(a[0], alp, color="red")
    plt.ylabel(r'Skewness, $\alpha$', fontsize=17)
    plt.xlabel("S1[phd]", fontsize=17)
    plt.show()

    # fit normalization
    norm_factor = np.sum(a[4])
    norm_counts = a[4]/norm_factor
    norm_popt, norm_pcov = curve_fit(FUNC_NORM, a[0], norm_counts,
                                     bounds=([-np.inf, -np.inf, -np.inf, -10], [np.inf, np.inf, np.inf, -1]))
    plt.scatter(a[0], norm_counts)
    domain = np.linspace(1.5, 100, 500)
    norms = FUNC_NORM(domain, norm_popt[0], norm_popt[1], norm_popt[2], norm_popt[3])
    norm_area = cumtrapz(norms, domain)
    print(norm_popt)
    plt.plot(domain, norms, color="red")
    plt.ylabel("Counts", fontsize=17)
    plt.xlabel("S1[phd]", fontsize=17)
    plt.text(5, 0.005, r"$N(S1)\approx \frac{0.055}{(S1+1.502) ^ {1.5}} - \frac{0.141}{S1^3+1.502}+0.01$", fontsize=15)
    plt.ylim(0, 0.015)
    plt.show()
    norm_popt = np.append(norm_popt, norm_area[-1])
    opts = np.asarray([eps_popt, omeg_popt, alp_popt, norm_popt])

    S1_act = data[0, :]
    S1_act = S1_act[S1_act<100][:10000]
    S1_pred = EReval.generateER(10000)[0]
    plt.hist(S1_act, alpha=0.8)
    plt.hist(S1_pred, alpha=0.8)
    plt.show()
    # print(anderson_ksamp([S1_act, S1_pred]))
    np.save("Data/ER_Fit/ER_popts", opts)  # saves interpolation parameters
    print(opts)




if __name__ == "__main__":
    main()
