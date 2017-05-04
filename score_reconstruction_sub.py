#!/usr/bin/env python
__author__ = 'jesse'
''' give this a wnid_graph and a reconstruction attempt (synsets and associated observations)

    matches reconstructed synsets to gold synsets
    prints recall/precision of observations at wnid level after alignment
    outputs pickled list of aligned gold wnids parallel to the input synsets; can contain None entries for non-aligned

'''

import argparse
import copy
import pickle
import sys


def main():

    syn_idx = FLAGS_syn_idx

    # read infiles
    print "reading in graph, observations, and reconstruction for syn_idx " + str(syn_idx) + "..."
    f = open(FLAGS_graph_infile, 'rb')
    wnids, _, nps, _ = pickle.load(f)
    f.close()
    f = open(FLAGS_wnid_obs_url_infile, 'rb')
    wnid_urls = pickle.load(f)
    f.close()
    f = open(FLAGS_reconstruction_infile, 'rb')
    re_synsets, re_syn_obs = pickle.load(f)
    f.close()
    f = open(FLAGS_np_train_obs, 'rb')
    np_train_observations = pickle.load(f)
    f.close()
    train_observations = []
    for np_idx in range(0, len(nps)):
        if np_idx in np_train_observations:
            train_observations.extend(np_train_observations[np_idx])
    print "... done"

    # cut synsets down to size by observing whether they actually have observations (urls)
    old_wnids = copy.deepcopy(wnids)
    wnids = [old_wnids[wnid_idx]
             for wnid_idx in range(0, len(old_wnids))
             if old_wnids[wnid_idx] in wnid_urls]

    syn_obs = [(wnids[syn_idx], obs_idx) for obs_idx in range(0, len(wnid_urls[wnids[syn_idx]]))
               if (wnids[syn_idx], obs_idx) in train_observations]
    # n = len(syn_obs)  # DEBUG
    row = []
    for re_idx in range(0, len(re_synsets)):
        row.append(sum([1 if obs in syn_obs else 0
                        for np_idx in re_synsets[re_idx]
                        for obs in re_syn_obs[(np_idx, re_idx)]]))
    print "calculated row with " + str(len(row)) + " entries summing to " + str(sum(row))

    # if n != sum(row):  # DEBUG
    #     sys.exit("FATAL: " + str(n) + " syn_obs with re row sum " + str(sum(row)))

    n = sum(row)  # DEBUG; this is only valid if reconstruction can drop invalid observations, which I think it can

    # write synsets, syn_obs of induced topology
    print "writing row and n to file..."
    f = open(FLAGS_outfile, 'wb')
    d = [row, n]
    pickle.dump(d, f)
    f.close()
    print "... done"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_infile', type=str, required=True,
                        help="wnid graph used to construct observations")
    parser.add_argument('--wnid_obs_url_infile', type=str, required=True,
                        help="wnid observations url file (faster to load; don't actually need numbers)")
    parser.add_argument('--reconstruction_infile', type=str, required=True,
                        help="reconstruction structures to be tested")
    parser.add_argument('--np_train_obs', type=str, required=True,
                        help="observations to consider")
    parser.add_argument('--syn_idx', type=int, required=True,
                        help="index of gold synset to build row of overlap with reconstruction for")
    parser.add_argument('--outfile', type=str, required=True,
                        help="write homogeneity, completeness, and v-measure list here")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
