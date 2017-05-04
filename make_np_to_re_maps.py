#!/usr/bin/env python
__author__ = 'jesse'

import argparse
import pickle


def find_re_synsets_with_np(np, nps, re_synsets):

    np_idx = nps.index(np)
    return [re_idx for re_idx in range(0, len(re_synsets)) if np_idx in re_synsets[re_idx]]


def main():

    # read infiles
    print "reading in graph, observations, and reconstruction..."
    with open(FLAGS_graph_infile, 'rb') as f:
        wnids, synsets, nps, polysems = pickle.load(f)
    re_synsets_list = []
    for fn in FLAGS_reconstruction_infiles.split(','):
        with open(fn, 'rb') as f:
            re_synsets, _ = pickle.load(f)
            re_synsets_list.append(re_synsets)
    print "... done"

    # Create structure of interest.
    print "creating map from nps to lists of their re_idxs in supplied reconstructions..."
    np_to_re = []
    for np in nps:
        np_to_re.append([find_re_synsets_with_np(np, nps, re_synsets)
                        for re_synsets in re_synsets_list])
    print "... done"

    # Print some stats about the data.
    print "num nps: " + str(len(nps))
    print "avg re_idxs per np: " + str([sum([len(np_to_re[np][idx]) for np in range(len(nps))]) / float(len(nps))
                                        for idx in range(len(re_synsets_list))])

    # Output mapping.
    print "writing map to file..."
    with open(FLAGS_outfile, 'wb') as f:
        pickle.dump([nps, np_to_re], f)
    print "... done"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_infile', type=str, required=True,
                        help="wnid graph used to construct observations")
    parser.add_argument('--reconstruction_infiles', type=str, required=True,
                        help="reconstruction structures")
    parser.add_argument('--outfile', type=str, required=True,
                        help="map outfile")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
