#!/usr/bin/env python
__author__ = 'jesse'
''' give this a wnid_graph and associated wnid_observations

    outputs a structure of np_observations by splitting observations among 

'''

import argparse
import pickle
import random
import sys


def main():

    # read infiles
    print "reading in graph and observations..."
    f = open(FLAGS_graph_infile, 'rb')
    wnids, synsets, _, _ = pickle.load(f)
    f.close()
    print "... read " + str(len(wnids)) + " wnids from graph"
    f = open(FLAGS_wnid_urls, 'rb')
    wnid_urls = pickle.load(f)
    f.close()
    print "... read " + str(len(wnid_urls.keys())) + " wnids with obs urls"
    try:
        f = open(FLAGS_missing_or_duplicate, 'rb')
        wnid_exclude = pickle.load(f)
        f.close()
        print "... read " + str(len(wnid_exclude.keys())) + " wnids with obs to exclude"
    except IOError:
        wnid_exclude = {}
        print "... WARNING: could not read missing or duplicated map"

    # divide observations evenly among each synset's nps
    print ("dividing observations evenly among wnid synsets' noun phrases and " +
           "collapsing observations for polysemous noun phrases...")
    np_to_f_idx = {}
    for wnid_idx in range(0, len(wnids)):
        wnid = wnids[wnid_idx]
        if wnid not in wnid_urls:
            continue

        new_order = range(0, len(wnid_urls[wnid]))
        random.shuffle(new_order)

        # pass through new ordering obs_idxs and remove those to be excluded
        if wnid in wnid_exclude:
            new_order = [obs_idx for obs_idx in new_order if obs_idx not in wnid_exclude[wnid]]

        synset = synsets[wnid_idx]
        obs_per_np = float(len(new_order)) / len(synset)
        head_obs = 0
        for np_idx_pos in range(0, len(synset)):
            np_idx = synset[np_idx_pos]
            if np_idx not in np_to_f_idx:
                np_to_f_idx[np_idx] = []
            tail_obs = head_obs + obs_per_np
            if tail_obs > len(new_order) or np_idx_pos == len(synset)-1:
                tail_obs = len(new_order)
            np_to_f_idx[np_idx].extend([(wnid, new_order[obs_idx])
                                        for obs_idx in range(int(head_obs), int(tail_obs))])
            head_obs = tail_obs
    print "... done; indexed feature observations for "+str(len(np_to_f_idx))+" noun phrases"

    # write np observations to file
    print "writing np to feature observation indices map to file..."
    f = open(FLAGS_outfile, 'wb')
    pickle.dump(np_to_f_idx, f)
    f.close()
    print "... done"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_infile', type=str, required=True,
                        help="wnid graph used when getting observations")
    parser.add_argument('--wnid_urls', type=str, required=True,
                        help="wnid urls used to index into distributed observations")
    parser.add_argument('--missing_or_duplicate', type=str, required=True,
                        help="wnid to obs_idx map of missing/duplicate entries to ignore")
    parser.add_argument('--outfile', type=str, required=True,
                        help="output pickles of valid nps->observation features maps")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
