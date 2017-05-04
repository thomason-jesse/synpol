#!/usr/bin/env python
__author__ = 'jesse'
''' give this a set of np_observations

    outputs synsets and observation pairing attempting to reconstruct original wnid_graph based on observations alone

'''

import argparse
import pickle
from gap_statistic_functions import *


def main():

    # read in pre-calculated information
    f = open(FLAGS_params_infile, 'rb')
    clusters, means, syn_idx, syn_jdx = pickle.load(f)
    f.close()

    # do the grunt work
    set_x = clusters[syn_idx]
    set_x = numpy.concatenate((set_x, clusters[syn_jdx][:]), 0)
    ref_sets = get_reference_sets(set_x)
    ks = [1, 2]
    mu_k1 = reevaluate_centers({0: set_x}, len(set_x[0]))
    mu_k2 = [means[syn_idx], means[syn_jdx]]
    clusters_k1 = {0: set_x}
    clusters_k2 = {0: clusters[syn_idx], 1: clusters[syn_jdx]}
    w_k1 = numpy.log(wk(mu_k1, clusters_k1))
    w_k2 = numpy.log(wk(mu_k2, clusters_k2))
    merge, gap, _ = gap_statistic(ks, [w_k1, w_k2], ref_sets)

    # write merge value and calculated gap to pickle
    f = open(FLAGS_outfile, 'wb')
    d = [merge, gap]
    pickle.dump(d, f)
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--params_infile', type=str, required=True,
                        help="parameters to pass to gap_statistic")
    parser.add_argument('--outfile', type=str, required=True,
                        help="output pickled return values of gap_statistic")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
