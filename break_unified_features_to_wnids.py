#!/usr/bin/env python
__author__ = 'jesse'
''' takes a map of wnids -> observations and writes pickles for each wnid -> observation entry

'''

import argparse
import pickle
import os
import time


def main():

    # read infiles
    print "reading in urls and unified features..."
    f = open(FLAGS_wnid_obs, 'rb')
    wnid_obs = pickle.load(f)
    f.close()
    f = open(FLAGS_wnid_urls, 'rb')
    wnid_urls = pickle.load(f)
    wnids = wnid_urls.keys()
    f.close()
    print "... done"

    # write individual wnid -> feature pickles (single entry maps)
    print "writing wnid -> observations to file..."
    for wnid_idx in range(0, len(wnids)):
        fname = str(wnid_idx) + "_" + FLAGS_outfile
        with open(fname, 'wb') as f:
            d = {wnids[wnid_idx]: wnid_obs[wnids[wnid_idx]]}
            pickle.dump(d, f)
    print "... done"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wnid_obs', type=str, required=True,
                        help="unified wnid to observations feature pickle")
    parser.add_argument('--wnid_urls', type=str, required=True,
                        help="wnid urls used to key into wnid idxs")
    parser.add_argument('--outfile', type=str, required=True,
                        help="suffix of output pickles of inidividual feature files")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
