#!/usr/bin/env python
__author__ = 'jesse'

import argparse
import os
import pickle


def write_map(m, outdir, t):

    for syn_idx in range(len(m)):
        # Create a directory for each syn_idx
        target_dir = os.path.join(outdir, str(syn_idx)+t)
        if not os.path.isdir(target_dir):
            os.system("mkdir " + target_dir)

        # Write a file for (np_idx, alt_idx) key
        for key in m[syn_idx]:
            np_idx, alt_idx = key
            fn = os.path.join(target_dir, str(np_idx)+"_"+str(alt_idx)+".pickle")
            if not os.path.isfile(fn):
                with open(fn, 'wb') as f:
                    pickle.dump(m[syn_idx][key], f)


def main():

    print "reading in mismatch maps..."
    f = open(FLAGS_maps_infile, 'rb')
    gold_maps, hyp_maps = pickle.load(f)
    f.close()
    print "... done"

    # Write individual files as output maps
    print "writing gold mismatch maps..."
    write_map(gold_maps, FLAGS_outdir, "gold")
    print ".. done"
    print "writing hyp mismatch maps..."
    write_map(hyp_maps, FLAGS_outdir, "hyp")
    print "... done"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--maps_infile', type=str, required=True,
                        help="maps to mismatches from get_mturk_pairs.py")
    parser.add_argument('--outdir', type=str, required=True,
                        help="directory to dump mismatch maps into")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
