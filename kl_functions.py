import numpy as np
import gap_statistic_functions


def multivariate_kl_divergence(u1, v1, u2, v2):

    log_v_ratio = np.log(v2 / v1)
    d = len(u1)  # trace of diagonal identity matrix in size of variance
    v2_inv = 1.0 / v2
    trace_var_product = np.sum(np.multiply(v2_inv, v1))
    diff_means = u2 - u1
    prod = np.multiply(np.multiply(diff_means, v2_inv), diff_means)
    return np.sum(0.5 * (log_v_ratio - d + trace_var_product + prod))


def multivariate_kl_distance_(u1, v1, u2, v2):
    return 0.5 * (multivariate_kl_divergence(u1, v1, u2, v2) + multivariate_kl_divergence(u2, v2, u1, v1))


def multivariate_kl_distance(u1, v1, u2, v2):

    if len(u1) == gap_statistic_functions.textf_size + gap_statistic_functions.imgf_size:
        img_u1 = u1[:gap_statistic_functions.imgf_size]
        img_u2 = u2[:gap_statistic_functions.imgf_size]
        img_v1 = v1[:gap_statistic_functions.imgf_size]
        img_v2 = v2[:gap_statistic_functions.imgf_size]
        img_d = multivariate_kl_distance_(img_u1, img_v1, img_u2, img_v2)

        text_u1 = u1[gap_statistic_functions.imgf_size:]
        text_u2 = u2[gap_statistic_functions.imgf_size:]
        text_v1 = v1[gap_statistic_functions.imgf_size:]
        text_v2 = v2[gap_statistic_functions.imgf_size:]
        text_d = multivariate_kl_distance_(text_u1, text_v1, text_u2, text_v2)

        return 0.5 * (img_d + text_d)

    else:
        return multivariate_kl_distance_(u1, v1, u2, v2)
