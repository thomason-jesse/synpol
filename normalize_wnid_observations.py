#!/usr/bin/env python
__author__ = 'jesse'
''' give this a wnid_graph and associated wnid_observations

    outputs normalized wnid_obs

'''

import argparse
import pickle
import random
from gap_statistic_functions import *


def main():

    # read infiles
    print "reading in graph and observations..."
    f = open(FLAGS_graph_infile, 'rb')
    wnids, synsets, nps, polysems = pickle.load(f)
    f.close()
    f = open(FLAGS_obs_infile, 'rb')
    wnid_observations = pickle.load(f)
    f.close()
    print "... done; read "+str(len(wnids))+" wnids from graph with "+str(len(wnid_observations)) + \
          " wnids associated with observations"

    # divide observations evenly among each synset's nps
    print "normalizing wnid obs"
    all_obs = []
    for wnid in wnid_observations:
        all_obs.extend(wnid_observations[wnid])
    mins, maxs = bounding_hypercube(all_obs)
    for wnid in wnid_observations:
        new_obs = [[(wnid_observations[wnid][idx][fdx] - mins[fdx]) / (maxs[fdx] - mins[fdx])
                    if maxs[fdx] - mins[fdx] > 0 else 0
                    for fdx in range(0, len(wnid_observations[wnid][idx]))]
                   for idx in range(0, len(wnid_observations[wnid]))]
        wnid_observations[wnid] = new_obs
    print "... done;"

    # write np observations to file
    print "writing normalized observations to file..."
    f = open(FLAGS_outfile, 'wb')
    pickle.dump(wnid_observations, f)
    f.close()
    print "... done"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_infile', type=str, required=True,
                        help="wnid graph used when getting observations")
    parser.add_argument('--obs_infile', type=str, required=True,
                        help="wnid observations file")
    parser.add_argument('--outfile', type=str, required=True,
                        help="output pickles of nps->observation features maps")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
