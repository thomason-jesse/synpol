#!/usr/bin/env python
__author__ = 'jesse'

import argparse
from gap_statistic_functions import *


def main():

    # read in obs
    with open(FLAGS_infile, 'rb') as f:
        num_k, obs = pickle.load(f)

    # do the grunt work
    km = KMeans(n_clusters=num_k, init=obs[:num_k], n_init=1)
    km.fit(obs)
    # print "......done"  # DONE
    mu = km.cluster_centers_
    clusters = cluster_points(obs, mu, cosine_distance)

    # write num_k, n_obs_classes to pickle
    with open(FLAGS_outfile, 'wb') as f:
        d = [mu, clusters]
        pickle.dump(d, f)

    print "mu: " + str(mu)  # DEBUG
    print "clusters: " + str(clusters)  # DEBUG


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, required=True,
                        help="pair of k and observations to cluster")
    parser.add_argument('--outfile', type=str, required=True,
                        help="pair of means and cluster assignments")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
