#!/usr/bin/env python
__author__ = 'jesse'
''' give this a np observations and a graph

    outputs synsets and observation pairing to reconstruct original graph as topline

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
    wnids, synsets, _, _ = pickle.load(f)
    f.close()
    f = open(FLAGS_wnid_obs_url_infile, 'rb')
    wnid_urls = pickle.load(f)
    f.close()
    f = open(FLAGS_np_obs_infile, 'rb')
    np_observations = pickle.load(f)
    f.close()
    print "... done"

    # cut synsets down to size by observing whether they actually have observations (urls)
    old_wnids = copy.deepcopy(wnids)
    old_synsets = copy.deepcopy(synsets)
    wnids = [old_wnids[wnid_idx]
             for wnid_idx in range(0, len(old_wnids))
             if old_wnids[wnid_idx] in wnid_urls]
    synsets = [old_synsets[wnid_idx]
               for wnid_idx in range(0, len(old_wnids))
               if old_wnids[wnid_idx] in wnid_urls]

    # instantiate synsets based on wnid_graph
    # populate syn_obs from wnid_observations
    print "instantiating observations given gold graph..."
    syn_obs = {}
    for syn_idx in range(0, len(synsets)):
        for np_idx in synsets[syn_idx]:
            key = (np_idx, syn_idx)
            if np_idx in np_observations:
                syn_obs[key] = [entry for entry in np_observations[np_idx] if entry[0] == wnids[syn_idx]]
            else:
                syn_obs[key] = []
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
    parser.add_argument('--wnid_obs_url_infile', type=str, required=True,
                        help="wnid observations url file (faster to load; don't actually need numbers)")
    parser.add_argument('--np_obs_infile', type=str, required=True,
                        help="np observations file")
    parser.add_argument('--outfile', type=str, required=True,
                        help="output pickled synsets and observations of gold graph")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
