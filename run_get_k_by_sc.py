#!/usr/bin/env python
__author__ = 'jesse'
''' give this a set of observations

    outputs results of get_k_by_gap_statistic to specified pickle

'''

import argparse
import pickle
import numpy
from sklearn.cluster import *
from gap_statistic_functions import collapse_small_clusters


def main():

    # read in obs
    f = open(FLAGS_obs_infile, 'rb')
    obs, n_clusters, alpha = pickle.load(f)
    f.close()

    # do the grunt work
    n_obs = numpy.asarray(obs)
    try:
        sc = SpectralClustering(n_clusters=min(n_clusters, len(n_obs)))
        n_obs_classes = sc.fit_predict(n_obs)
        num_k = len(set(n_obs_classes))
        if FLAGS_trim_poly == 1 and n_obs_classes is not None and num_k > 0:
            n_obs_classes, num_k = collapse_small_clusters(n_obs, n_obs_classes, num_k)
    except ValueError:  # Too few observations
        n_obs_classes = [0 for _ in range(0, len(n_obs))]
        num_k = 1

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
                        help="whether to trim polysemy sets to be greater than 1")
    parser.add_argument('--outfile', type=str, required=True,
                        help="output pickled return values from function")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
