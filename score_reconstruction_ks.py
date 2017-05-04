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

    syn_idx = FLAGS_syn_idx

    # read infiles
    print "reading in graph, observations, and reconstruction for syn_idx " + str(syn_idx) + "..."
    f = open(FLAGS_reconstruction_infile, 'rb')
    re_synsets, re_syn_obs = pickle.load(f)
    f.close()
    f = open(FLAGS_member_matrix, 'rb')
    a, n = pickle.load(f)
    f.close()
    print "... done"

    h_k_s_part = 0
    syn_members_in_other_re = sum([a[syn_idx][re_jdx] for re_jdx in range(0, len(re_synsets))])
    for re_idx in range(0, len(re_synsets)):
        syn_members_in_re = a[syn_idx][re_idx]
        if syn_members_in_re > 0:
            h_k_s_part += (syn_members_in_re / float(n)) * math.log(syn_members_in_re / float(syn_members_in_other_re))
    print "calculated h_k_s_part " + str(h_k_s_part)

    # write synsets, syn_obs of induced topology
    print "writing h_k_s_part to file..."
    f = open(FLAGS_outfile, 'wb')
    d = h_k_s_part
    pickle.dump(d, f)
    f.close()
    print "... done"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reconstruction_infile', type=str, required=True,
                        help="reconstruction structures to be tested")
    parser.add_argument('--member_matrix', type=str, required=True,
                        help="member matrix a and number of instances n")
    parser.add_argument('--syn_idx', type=int, required=False,
                        help="syn_idx to calculate re_idxs over")
    parser.add_argument('--outfile', type=str, required=True,
                        help="write h_k_s contribution here")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
