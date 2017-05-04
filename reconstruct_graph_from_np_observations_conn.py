#!/usr/bin/env python
__author__ = 'jesse'

import argparse
import pickle
from gap_statistic_functions import cosine_distance
from kl_functions import multivariate_kl_distance
import numpy
import sys


def main():

    try:
        penalty = FLAGS_penalty
    except:
        penalty = 2

    # read pickle inputs
    with open(FLAGS_synsets, 'rb') as f:
        synsets = pickle.load(f)
    with open(FLAGS_means, 'rb') as f:
        means = pickle.load(f)
    try:
        with open(FLAGS_vars) as f:
            vars = pickle.load(f)
    except:
        vars = None

    # copy param inputs
    syn_idx = FLAGS_syn_idx

    np_idx = synsets[syn_idx][0]  # single np per sense at this stage
    dist_row = {}
    for syn_jdx in range(syn_idx+1, len(synsets)):
        np_jdx = synsets[syn_jdx][0]

        if vars is None:
            dist = cosine_distance(means[syn_idx], means[syn_jdx])
        else:
            dist = multivariate_kl_distance(means[syn_idx], vars[syn_idx],
                                            means[syn_jdx], vars[syn_jdx])
            if not numpy.isfinite(dist):
                dist = float(sys.maxint)

        # induced senses of the same noun phrase should be collapsed last if at all
        # use penalty distance of 2, the maximum cosine distance
        if np_idx == np_jdx:
            dist += penalty

        # add entry to connectivity matrix with distance + penalty
        dist_row[syn_jdx] = dist

    # write mean to pickle
    f = open(FLAGS_outfile, 'wb')
    d = dist_row
    pickle.dump(d, f)
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--synsets', type=str, required=True,
                        help="synsets across which to form connectivity matrix")
    parser.add_argument('--means', type=str, required=True,
                        help="synset means")
    parser.add_argument('--vars', type=str, required=False,
                        help="synset variances")
    parser.add_argument('--syn_idx', type=int, required=True,
                        help="synset for which to calculate a mean")
    parser.add_argument('--penalty', type=int, required=False,
                        help="penalty distance for known split senses")
    parser.add_argument('--outfile', type=str, required=True,
                        help="output pickled return values from function")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
