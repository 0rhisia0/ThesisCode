import functions as fn
import constants as const
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
from scipy import integrate
BULK = 5600 * 1000
Emin = 1 * const.keV
Emax = fn.max_recoil_energy()
del_Er = 1
E_r = np.arange(Emin, Emax, del_Er)


# Finds log likelihood of poisson component of WIMP events for S1, S2 data
def pois_like_signal(mass, xsec, num_events, A=const.AXe, E_thr=6*const.keV):
    WIMP = [mass*const.GeV, (10**xsec)*const.cm2]
    E_ev, int_rate = fn.integrate_rate(E_r, WIMP, A)
    idx = find_nearest_idx(E_ev, E_thr)
    pred_events = int_rate[idx] * BULK
    poisson_prob = poisson.logpmf(num_events, pred_events)
    return poisson_prob


# Finds log likelihood of poisson component of WIMP events for S1, S2 data
def pois_like(mass, xsec, num_events, pred_ER, A=const.AXe, E_thr=6*const.keV):
    WIMP = [mass*const.GeV, (10**xsec)*const.cm2]
    E_ev, int_rate = fn.integrate_rate(E_r, WIMP, A)
    idx = find_nearest_idx(E_ev, E_thr)
    pred_NR = int_rate[idx] * BULK
    poisson_prob = poisson.logpmf(num_events, pred_NR + pred_ER)
    return poisson_prob, pred_NR



def find_indices(E_r, events):
    new_events = np.zeros(len(events))
    for i in range(len(events)):
        idx = find_nearest_idx(E_r, events[i])
        new_events[i] = idx
    return new_events


def find_nearest_idx(array, value):
    array = array - value
    idx = (np.abs(array)).argmin()
    return idx


""" DEPRECATED FUNCTIONS DO NOT CALL"""

def events_likelihood(E_r, events, WIMP, A, E_thr, del_Er):
    """
    Finds the log (base e) likelihood of events given WIMP and target parameters in energy space
    """
    obs_events = len(events)  # num events observed
    E_r, int_rate = fn.integrate_rate(E_r, WIMP, A)  # integrated rate
    idx = find_nearest_idx(E_r, E_thr)
    pred_events = int_rate[idx]*BULK  # multiplied by kg*days of detector chosen

    poisson_prob = poisson.logpmf(obs_events, pred_events)  # poisson probability of total event number
    e_prob = energy_prob(events, WIMP, A, E_r, del_Er)  # product probability of event energies

    return poisson_prob + e_prob


def energy_prob(events, WIMP, A, E_r, del_Er):
    """
    calculates the product probability of events given WIMP and event parameters
    """
    if not len(events):
        return 1  # empty product
    dif_rate = fn.diff_rate(E_r, WIMP, A)
    events = events - 1  # converting energies into indices by subtracting threshold away from it
    E_r = E_r[:60]
    weights = 1 / (1 + np.exp(-1.35 * E_r + 8.1))
    dif_rate = dif_rate[:60]
    dif_rate *= weights
    dif_rate /= np.sum(dif_rate)
    prob = 0
    for event in events:
        prob += np.log(dif_rate[int(event)])  # product probability as ln sum
    return prob