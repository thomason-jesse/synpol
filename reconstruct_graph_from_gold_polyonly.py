#!/usr/bin/env python
__author__ = 'jesse'
''' give this a np observations and a graph

    outputs synsets and observation pairing to reconstruct original graph senses (no synonymy) as topline

'''

import argparse
import pickle
import operator
import scipy.cluster.vq as sci_cluster
import numpy
from scipy.spatial.distance import cosine
import copy
import random


def main():

    # read infiles
    print "reading in graph and observations..."
    f = open(FLAGS_graph_infile, 'rb')
    _, synsets, _, _ = pickle.load(f)
    f.close()
    f = open(FLAGS_np_obs_infile, 'rb')
    np_observations = pickle.load(f)  # keys are np_idxs, values are (wnid, obs_idx) key pairs
    f.close()
    print "... done"

    # instantiate synsets and syn_obs from np_observations
    print "instantiating observations given gold graph..."
    syn_obs = {}  # keys are (np_idx, syn_idx), values are entries in np_observations
    synsets = []  # values are lists of np_idxs
    wnid_np_pairs = []  # keys into synsets
    for np_idx in np_observations:
        for wnid, obs_idx in np_observations[np_idx]:
            wnid_np = (wnid, np_idx)
            if wnid_np not in wnid_np_pairs:
                wnid_np_pairs.append(wnid_np)
                synsets.append([])
            syn_idx = wnid_np_pairs.index(wnid_np)
            if np_idx not in synsets[syn_idx]:
                synsets[syn_idx].append(np_idx)
            key = (np_idx, syn_idx)
            if key not in syn_obs:
                syn_obs[key] = []
            syn_obs[key].append((wnid, obs_idx))
    print "... done"
    print "|synsets|="+str(len(synsets))  # DEBUG
    print "|syn_obs|="+str(len(syn_obs.keys()))  # DEBUG

    # write synsets, syn_obs of induced topology
    print "writing gold synsets and observation map to file..."
    f = open(FLAGS_outfile, 'wb')
    d = [synsets, syn_obs]
    pickle.dump(d, f)
    f.close()
    print "... done"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_infile', type=str, required=True,
                        help="wnid graph used to construct observations")
    parser.add_argument('--np_obs_infile', type=str, required=True,
                        help="np observations file")
    parser.add_argument('--outfile', type=str, required=True,
                        help="output pickled synsets and observations of gold graph")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
