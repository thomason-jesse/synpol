#!/usr/bin/env python
__author__ = 'jesse'
''' produces a subset of np observations that participate in a valid polysemous relationship

'''

import argparse
import copy
import math
import pickle
import os
import sys
import time


def main():

    # read infiles
    print "reading in graph, observations, and reconstruction..."
    f = open(FLAGS_graph_infile, 'rb')
    wnids, synsets, nps, polysems = pickle.load(f)
    f.close()
    f = open(FLAGS_wnid_obs_url_infile, 'rb')
    wnid_urls = pickle.load(f)
    f.close()
    f = open(FLAGS_np_train_obs, 'rb')
    np_train_observations = pickle.load(f)
    f.close()
    print "... done"

    print "cutting observations to only nps that are polysemous with wnids that have url obs"
    trimmed_obs = {}
    for np_idx in np_train_observations:
        if np_idx in polysems:
            true_pol = 0
            for wnid in polysems[np_idx]:
                if wnid in wnid_urls:
                    true_pol += 1
                    if true_pol > 1:
                        break
            if true_pol > 1:
                trimmed_obs[np_idx] = np_train_observations[np_idx]
    print "... done; trimmed from " + str(len(np_train_observations)) + " nps to " + str(len(trimmed_obs)) + " nps"

    # write synsets, syn_obs of induced topology
    print "writing reconstructed to gold wnid mapping and stats to file..."
    f = open(FLAGS_outfile, 'wb')
    d = trimmed_obs
    pickle.dump(d, f)
    f.close()
    print "... done"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_infile', type=str, required=True,
                        help="wnid graph used to construct observations")
    parser.add_argument('--wnid_obs_url_infile', type=str, required=True,
                        help="wnid observations url file")
    parser.add_argument('--np_train_obs', type=str, required=True,
                        help="observations to consider")
    parser.add_argument('--outfile', type=str, required=True,
                        help="write np observation subset")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
