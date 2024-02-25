# metrics
from itertools import combinations

import numpy as np
from fairlearn.reductions import BoundedGroupLoss, ZeroOneLoss
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, \
    mean_squared_error, r2_score


# explicit function to normalize array
def normalize(arr, t_min, t_max, v_min, v_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = v_max - v_min
    for i in arr:
        temp = (((i - min(arr)) * diff) / diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr


def normalisation(arr, t_min, t_max):
    norm_arr = []
    diff_arr = t_max - t_min
    for i in arr:
        temp = (i - t_min) / diff_arr
        norm_arr.append(temp)
    return norm_arr


def compute_metrics(y, y_pred):
    return {
        'evs': explained_variance_score(y, y_pred),
        'mxe': max_error(y, y_pred),
        'mae': mean_absolute_error(y, y_pred),
        'mse': mean_squared_error(y, y_pred),
        'r2s': r2_score(y, y_pred)
    }


# Price of fairness
def pof(mse_non_mitigated, mse_mitigated):
    return mse_mitigated / mse_non_mitigated


# Agarwal2019 - Statistical parity using KS statistic - SP-KS
# SP - KS
def sp_ks(y_pred, y_pred_priv, y_pred_unpr):
    ks_priv = stats.kstest(y_pred_priv, y_pred)
    ks_unpr = stats.kstest(y_pred_unpr, y_pred)
    return ks_unpr.statistic - ks_priv.statistic


def compute_sp_ks(y_pred, classes, value_privileged):
    y_priv = []
    y_unpr = []
    for e in range(len(y_pred)):
        if classes[e] != value_privileged:
            y_unpr.append(y_pred[e])
        else:
            y_priv.append(y_pred[e])
    return sp_ks(y_pred, y_priv, y_unpr)


# Yan et all
# SP1 - sp_mi  MI(Y') - MI(Y)
# SP2 - sp_avg_outcome Is measured by the distance of avg outcomes of each sensitive group.
# EA - equal accuracy - Is measured by the distance of MAE across different groups. Negative values

# SP1 MI(Y') - MI(Y)
# random_state = 5 to reproducible results
def sp_mi(X, y, y_pred, index):
    mi_y = mutual_info_regression(X, y, random_state=5)
    mi_y_pred = mutual_info_regression(X, y_pred, random_state=5)
    sp = mi_y_pred - mi_y
    return sp[index]


# SP2 Is measured by the distance of avg outcomes of each sensitive group. Negative values.
def sp_avg_outcome(y_predicted_priv, y_predicted_unpriv):
    avg_outcome_priv = np.average(y_predicted_priv)
    avg_outcome_unpr = np.average(y_predicted_unpriv)
    return avg_outcome_unpr - avg_outcome_priv


def compute_sp_avg_outcome(y_pred, classes, value_privileged):
    y_priv = []
    y_unpr = []
    for e in range(len(y_pred)):
        if classes[e] != value_privileged:
            y_unpr.append(y_pred[e])
        else:
            y_priv.append(y_pred[e])
    return sp_avg_outcome(y_priv, y_unpr)


# EA Is measured by the distance of MAE across different groups. Negative values
def ea(y_priv, y_pred_priv, y_unpr, y_pred_unpr):
    mae_priv = mean_absolute_error(y_priv, y_pred_priv)
    mae_unpr = mean_absolute_error(y_unpr, y_pred_unpr)
    return mae_unpr - mae_priv


def compute_ea(y, y_pred, classes, value_privileged):
    y_priv = []
    y_unpr = []
    y_pred_priv = []
    y_pred_unpr = []

    for e in range(len(y_pred)):
        if classes[e] != value_privileged:
            y_unpr.append(y[e])
            y_pred_unpr.append(y_pred[e])
        else:
            y_priv.append(y[e])
            y_pred_priv.append(y_pred[e])
    return ea(y_priv, y_pred_priv, y_unpr, y_pred_unpr)


# Aghaei - DIDI - Disparate Impact discrimination Index
def didi_protected(y_protected, y):
    # print(sum(y_protected)/len(y_protected))
    # print(1/len(y) * sum(y))
    return abs(sum(y_protected) / len(y_protected) - 1 / len(y) * sum(y))


# computed over predicted values
def didi(y, classes, value_privileged, v_min, v_max):
    y_priv = []
    y_unpr = []
    # required to didi
    # y_norm = normalize(y, -1, 1, v_min, v_max)
    for e in range(len(y)):
        if classes[e] != value_privileged:
            y_unpr.append(y[e])
        else:
            y_priv.append(y[e])
    return didi_protected(y_priv, y) + didi_protected(y_unpr, y)


# bgl fairlearn
def bgl_fairlearn(X, y_true, y_pred, sensitive_features):
    bgl = BoundedGroupLoss(ZeroOneLoss(), upper_bound=0.1)
    bgl.load_data(X, y_true, sensitive_features=sensitive_features)
    print(bgl.gamma(lambda X: y_pred))


#

# General binary class ratio calculation
def ratio(p, q):
    assert p.shape[1] == 2
    assert q.shape[1] == 2

    r_i = p[:, 1] / p[:, 0] * q[:, 0] / q[:, 1]
    r = np.mean(r_i)

    return r


#
# Conditional Mutual Information Measures
#

# Conditional Mutual Information, probabilistic
def cmi_prob(p, q, normalise=True):
    N = p.shape[0]
    mi_n = p * (np.log(p) - np.log(q))
    mi = mi_n.sum() / N
    mi = mi / cent_prob(q) if normalise else mi
    return mi


# Conditional Mutual Information, empirical
def cmi_emp(p, q, a, normalise=True):
    N = len(a)
    ind = np.arange(N)
    p_emp, q_emp = p[ind, a], q[ind, a]
    if normalise:
        mi = 1. - np.mean(np.log(p_emp)) / np.mean(np.log(q_emp))
    else:
        mi = np.mean(np.log(p_emp) - np.log(q_emp))
    return mi


# Empirical or Probabilistic approaches depending on args
def cmi(p, q, a=None, normalise=True):
    if a is None:
        mi = cmi_prob(p, q, normalise)
    else:
        mi = cmi_emp(p, q, a, normalise)
    return mi


# Conditional entropy normaliser, probabilistic
def cent_prob(p):
    N = p.shape[0]
    h_i = - p * np.log(p)
    h = h_i.sum() / N
    return h


def compute_mi_measures(p_a, p_y, p_s, p_sy, a=None, normalise=True):
    ind = cmi(p_s, p_a, a, normalise)
    sep = cmi(p_sy, p_y, a, normalise)
    suf = cmi(p_sy, p_s, a, normalise)
    est_type = "est" if a is None else "emp"
    print(f"NMI {est_type} -- Ind: {ind:.4f}, Sep: {sep:.4f}, Suf: {suf:.4f}")
    return ind, sep, suf


def compute_er_measures(p_a, p_y, p_s, p_sy):
    ind = ratio(p_s, p_a)
    sep = ratio(p_sy, p_y)
    suf = ratio(p_sy, p_s)
    print(f"E[r] -- Ind: {ind:.4f}, Sep: {sep:.4f}, Suf: {suf:.4f}")
    return ind, sep, suf


def discretization(y, A, max_segments=100):
    YD = None  # normal
    for n_segments in np.arange(max_segments, 0, -1):
        YD = np.digitize(y, np.linspace(y.min(), y.max(), n_segments + 1)[1:-1])
        check_valid = 0
        for a in set(A):
            if len(set(YD[A == a])) != n_segments:
                break
            check_valid += 1

        if check_valid == len(set(A)):
            break
    return YD


def compute_deo(y, p, A):
    yd = discretization(y, A)
    stack = []
    for yy in set(yd):
        yy_stack = []
        for a, b in combinations(set(A), 2):
            Iay = np.logical_and(A == a, yd == yy)
            Iby = np.logical_and(A == b, yd == yy)
            if sum(Iay) > 0 and sum(Iby) > 0:
                yy_stack.append(
                    np.square(np.subtract(p[Iay].mean(), p[Iby].mean())))  # normal
                #print(p[Iay], p[Iby])
                #print('las medias', p[Iay].mean(), p[Iby].mean())
        if len(yy_stack) > 0:
            stack.append(np.mean(yy_stack))
        #print(stack)
        #print('el valor', np.sqrt(np.mean(stack)))
    return np.sqrt(np.mean(stack))
