import numpy as np


def calibrate_isotonic(cal_labels, cal_probs, new_probs):
    """
    Apply isotonic regression calibration.
    """
    cal_labels = np.asarray(cal_labels, dtype=int)
    cal_probs = np.asarray(cal_probs, dtype=float)
    new_probs = np.asarray(new_probs, dtype=float)

    sorted_indices = np.argsort(cal_probs)
    sorted_labels = cal_labels[sorted_indices]
    sorted_probs = cal_probs[sorted_indices]

    # each entry is (mean, count)
    # example
    # [0.1,0.3,0.7,0.9]
    # [  0,  1,  0,  1]
    groups = []
    for label in sorted_labels:
        groups.append([label, 1])
        if len(groups) >= 2 and groups[-2][0] > groups[-1][0]:
            p_mean, p_cnt = groups.pop(-2)
            c_mean, c_cnt = groups.pop(-1)

            cnt = p_cnt + c_cnt
            mean = (p_mean * p_cnt + c_mean * c_cnt) / cnt
            groups.append([mean, cnt])

    calibrated = []
    for mean, cnt in groups:
        calibrated.extend([mean] * cnt)

    calibrated = np.asarray(calibrated, dtype=float)
    interp = np.interp(new_probs, sorted_probs, calibrated).tolist()

    return interp
    