#!/usr/bin/env python
__author__ = 'jesse'

import argparse
import pickle
from gap_statistic_functions import *


def main():

    try:
        fixed_k_val = FLAGS_fixed_k_val
    except:
        fixed_k_val = None

    try:
        start_k_val = FLAGS_start_k_val
    except:
        start_k_val = None

    try:
        max_size = FLAGS_max_size
    except:
        max_size = sys.maxint

    n_ref_sets = FLAGS_ref_sets

    # read in obs
    f = open(FLAGS_obs_infile, 'rb')
    obs = pickle.load(f)
    f.close()
    f = open(FLAGS_conn_infile, 'rb')
    conn = pickle.load(f)
    f.close()

    # read in means / variances
    try:
        f = open(FLAGS_vars_infile, 'rb')
        variances = pickle.load(f)
        f.close()
        print "using kl distance"
    except:
        print "not using kl distance"
        variances = None

    # do the grunt work
    n_obs = numpy.asarray(obs)
    n_conn = numpy.asarray(conn)
    if fixed_k_val is None:
        num_k, n_obs_classes = get_k_by_gap_statistic_agg(n_obs, n_conn, n_ref_sets, start_k=start_k_val)
    else:

        agg = hac(n_clusters=fixed_k_val, affinity='precomputed',
                  connectivity=n_conn, linkage='average', max_size=max_size,
                  means=n_obs, variances=variances)
        label_generator = agg.fit(n_obs)
        n_obs_classes = None
        print "getting yield from agg " + str(len(n_obs)-fixed_k_val) + " times"  # DEBUG
        for _ in range(0, len(n_obs)-fixed_k_val):
            n_obs_classes = label_generator.next()
        num_k = len(set(n_obs_classes))
        print "got " + str(num_k) + " clusters"  # DEBUG

    # write num_k, n_obs_classes to pickle
    f = open(FLAGS_outfile, 'wb')
    d = [num_k, n_obs_classes]
    pickle.dump(d, f)
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--obs_infile', type=str, required=True,
                        help="observations to operate over")
    parser.add_argument('--conn_infile', type=str, required=True,
                        help="precomputed distances between observations")
    parser.add_argument('--vars_infile', type=str, required=False,
                        help="variances of observation clsuters for kl distances")
    parser.add_argument('--ref_sets', type=int, required=True,
                        help="number of reference sets to use during gap statistic")
    parser.add_argument('--fixed_k_val', type=int, required=False,
                        help='fixed num clusters; will not use gap statistic to estimate')
    parser.add_argument('--start_k_val', type=int, required=False,
                        help='start num clusters; use gap statistic to estimate lower')
    parser.add_argument('--max_size', type=int, required=False,
                        help='maximum cluster membership')
    parser.add_argument('--outfile', type=str, required=True,
                        help="output pickled return values from function")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
