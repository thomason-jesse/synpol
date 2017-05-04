#!/usr/bin/env python
__author__ = 'jesse'
''' give this a set of observations

    outputs results of get_k_by_gap_statistic to specified pickle

'''

import argparse
import pickle
from gap_statistic_functions import *


def main():

    try:
        with open(FLAGS_init_obs_infile, 'rb') as f:
            init_obs = pickle.load(f)
        with open(FLAGS_init_conn_infile, 'rb') as f:
            init_conn = pickle.load(f)
        print "read in " + str(len(init_obs)) + " buckshot obs of size " + str(len(init_obs[0]))
        buckshot = True
    except (TypeError, NameError):
        init_obs = init_conn = None
        buckshot = False

    min_k = FLAGS_min_k

    # read in obs
    try:
        with open(FLAGS_obs_infile, 'rb') as f:
            obs, _, _ = pickle.load(f)
    except ValueError:
        with open(FLAGS_obs_infile, 'rb') as f:
            obs = pickle.load(f)

    # do the grunt work
    n_obs = numpy.asarray(obs)
    if len(n_obs) >= FLAGS_trim_poly * 2:  # at least two minimally-sized clusters might exist
        if buckshot:
            buckshot_info = [init_obs, init_conn]
        else:
            buckshot_info = None
        num_k, n_obs_classes = get_k_by_gap_statistic(n_obs, FLAGS_trim_poly, min_k, dist="cosine",
                                                      buckshot=buckshot_info)
        if num_k is None or n_obs_classes is None:
            num_k = 1
            n_obs_classes = [0 for _ in range(0, len(n_obs))]
        if FLAGS_trim_poly > 1 and n_obs_classes is not None and num_k > 0:
            n_obs_classes, num_k = collapse_small_clusters(n_obs, n_obs_classes, num_k,
                                                           FLAGS_trim_poly, dist="cosine")
    else:
        num_k = 1
        n_obs_classes = [0 for _ in range(0, len(n_obs))]

    # write num_k, n_obs_classes to pickle
    f = open(FLAGS_outfile, 'wb')
    d = [num_k, n_obs_classes]
    pickle.dump(d, f)
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--obs_infile', type=str, required=True,
                        help="observations to operate over")
    parser.add_argument('--trim_poly', type=int, required=True,
                        help="whether to trim polysemy sets to be greater than given value")
    parser.add_argument('--init_obs_infile', type=str, required=False,
                        help="observations for agglomerative initialization of cluster means")
    parser.add_argument('--init_conn_infile', type=str, required=False,
                        help="connectivity matrix for agglomerative initialization of cluster means")
    parser.add_argument('--min_k', type=int, required=True,
                        help="k to start with for gap statistic")
    parser.add_argument('--outfile', type=str, required=True,
                        help="output pickled return values from function")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
