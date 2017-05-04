#!/usr/bin/env python
__author__ = 'jesse'
''' give this a wnid_graph and associated wnid_observations

    outputs wnid observations with only entries for those in the given graph

'''

import argparse
import pickle


def main():

    # read infiles
    print "reading in graph and observations..."
    f = open(FLAGS_graph_infile, 'rb')
    wnids, _, _, _ = pickle.load(f)
    f.close()
    f = open(FLAGS_obs_infile, 'rb')
    wnid_observations_all = pickle.load(f)
    f.close()
    f = open(FLAGS_url_infile, 'rb')
    urls_all = pickle.load(f)
    f.close()
    print "... done"

    # create subset of observations corresponding to graph
    print "creating subset of wnid observations given graph..."
    wnid_observations = {}
    urls = {}
    for wnid in wnids:
        if wnid in wnid_observations_all:
            wnid_observations[wnid] = wnid_observations_all[wnid]
        if wnid in urls_all:
            urls[wnid] = urls_all[wnid]
    print "... done"

    # write wnid observations to file
    print "writing wnid observations to file..."
    f = open(FLAGS_obs_outfile, 'wb')
    pickle.dump(wnid_observations, f)
    f.close()
    f = open(FLAGS_url_outfile, 'wb')
    pickle.dump(urls, f)
    f.close()
    print "... done"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_infile', type=str, required=True,
                        help="wnid graph used when getting observations")
    parser.add_argument('--obs_infile', type=str, required=True,
                        help="wnid observations file")
    parser.add_argument('--url_infile', type=str, required=True,
                        help="wnid url observations file")
    parser.add_argument('--obs_outfile', type=str, required=True,
                        help="output subset wnid observations file")
    parser.add_argument('--url_outfile', type=str, required=True,
                        help="output subset wnid urls file")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
