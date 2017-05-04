#!/usr/bin/env python
__author__ = 'jesse'
''' give this a wnid_graph and a reconstruction attempt (synsets and associated observations)

    matches reconstructed synsets to gold synsets
    prints recall/precision of observations at wnid level after alignment
    outputs pickled list of aligned gold wnids parallel to the input synsets; can contain None entries for non-aligned

'''

import argparse
import copy
import math
import pickle
import os
import time


def main():

    re_idx = FLAGS_re_idx

    # read infiles
    print "reading in graph, observations, and reconstruction for re_idx " + str(re_idx) + "..."
    f = open(FLAGS_graph_infile, 'rb')
    wnids, synsets, _, _ = pickle.load(f)
    f.close()
    f = open(FLAGS_wnid_obs_url_infile, 'rb')
    wnid_urls = pickle.load(f)
    f.close()
    f = open(FLAGS_member_matrix, 'rb')
    a, n = pickle.load(f)
    f.close()
    print "... done"

    # cut synsets down to size by observing whether they actually have observations (urls)
    old_wnids = copy.deepcopy(wnids)
    old_synsets = copy.deepcopy(synsets)
    synsets = [old_synsets[wnid_idx]
               for wnid_idx in range(0, len(old_wnids))
               if old_wnids[wnid_idx] in wnid_urls]

    h_s_k_part = 0
    re_members_in_other_syn = sum([a[syn_jdx][re_idx] for syn_jdx in range(0, len(synsets))])
    for syn_idx in range(0, len(synsets)):
        re_members_in_syn = a[syn_idx][re_idx]
        if re_members_in_syn > 0:
            h_s_k_part += (re_members_in_syn / float(n)) * math.log(re_members_in_syn / float(re_members_in_other_syn))
    print "calculated h_s_k part " + str(h_s_k_part)

    # write synsets, syn_obs of induced topology
    print "writing h_s_k_part to file..."
    f = open(FLAGS_outfile, 'wb')
    d = h_s_k_part
    pickle.dump(d, f)
    f.close()
    print "... done"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_infile', type=str, required=True,
                        help="wnid graph used to construct observations")
    parser.add_argument('--wnid_obs_url_infile', type=str, required=True,
                        help="wnid observations url file (faster to load; don't actually need numbers)")
    parser.add_argument('--member_matrix', type=str, required=True,
                        help="member matrix a and number of instances n")
    parser.add_argument('--re_idx', type=int, required=False,
                        help="re_idx to calculate syn_idxs over")
    parser.add_argument('--outfile', type=str, required=True,
                        help="write h_s_k contribution here")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
