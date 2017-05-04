#!/usr/bin/env python
__author__ = 'jesse'
''' takes a wnid graph, a wnid -> observation map, a wnid -> textf observation map, a target wnid_idx, and
    a range of idxs to check for duplicates in; returns map from wnids to duplicate observation idxs

'''

import argparse
import numpy as np
import pickle
import sys


def main():

    # get target idx and range
    wnid_idx = FLAGS_target_wnid_idx
    wnid_idx_range = range(FLAGS_wnid_idx_start, FLAGS_wnid_idx_end)
    distributed = True if FLAGS_distributed == 1 else False

    # read infiles
    print "reading in urls and observations..."
    f = open(FLAGS_urls_infile, 'rb')
    wnid_urls = pickle.load(f)
    wnids = wnid_urls.keys()
    f.close()
    print "... read urls"
    if not distributed:
        f = open(FLAGS_obs_infile, 'rb')
        wnid_observations = pickle.load(f)
        f.close()
        print "... read image observations"
        f = open(FLAGS_textf_infile, 'rb')
        wnid_textf = {wnids[wnid_idx]: pickle.load(f)}
        f.close()
        print "... read text features"
    try:
        f = open(FLAGS_known_duplicates, 'rb')
        known_duplicates = pickle.load(f)
        f.close()
        print "... read known duplicates"
    except (IOError, EOFError):
        known_duplicates = {}
    print "... done"

    # detect duplicates based on feature matches
    print "detecting duplicate observations..."
    duplicates = {}  # indexed by wnid, value list of obs_idx that are duplicated elsewhere
    missing = {}

    # build np matrix of observations against which to compare for each wnid_jdx
    for wnid_jdx in wnid_idx_range:
        print "... in wnid_jdx " + str(wnid_jdx)  # DEBUG
        imgf_rows = []
        textf_rows = []
        row_wnid_obs_keys = []

        if distributed:
            try:
                with open(str(wnid_jdx) + "_" + FLAGS_obs_infile) as f:
                    wnid_observations = pickle.load(f)
                with open(str(wnid_jdx) + "_" + FLAGS_textf_infile) as f:
                    wnid_textf = {wnids[wnid_jdx]: pickle.load(f)}
            except (IOError, EOFError):
                print "... WARNING: no pickle for jdx " + str(wnid_jdx)  # DEBUG
                continue

        if wnids[wnid_jdx] not in wnid_observations:
            continue

        for obs_jdx in range(0, len(wnid_observations[wnids[wnid_jdx]])):

            if wnids[wnid_jdx] in known_duplicates and obs_jdx in known_duplicates[wnids[wnid_jdx]]:
                continue

            row_wnid_obs_keys.append((wnid_jdx, obs_jdx))

            alt_imgf = wnid_observations[wnids[wnid_jdx]][obs_jdx]
            alt_imgf_n = np.asarray(alt_imgf)
            if np.linalg.norm(alt_imgf) > 0:
                alt_imgf_n /= np.linalg.norm(alt_imgf)
            imgf_rows.append(alt_imgf_n)

            alt_textf = wnid_textf[wnids[wnid_jdx]][obs_jdx]
            alt_textf_n = np.asarray(alt_textf)
            if np.linalg.norm(alt_textf) > 0:
                alt_textf_n /= np.linalg.norm(alt_textf)
            textf_rows.append(alt_textf_n)

        imgf_m = np.asmatrix(imgf_rows)
        textf_m = np.asmatrix(textf_rows)

        if len(row_wnid_obs_keys) == 0:
            continue

        if distributed:
            try:
                with open(str(wnid_idx) + "_" + FLAGS_obs_infile) as f:
                    wnid_observations = pickle.load(f)
                with open(str(wnid_idx) + "_" + FLAGS_textf_infile) as f:
                    wnid_textf = {wnids[wnid_idx]: pickle.load(f)}
            except (IOError, EOFError):
                print "... FATAL: no pickle for target idx " + str(wnid_idx)  # DEBUG
                sys.exit()

        for obs_idx in range(0, len(wnid_observations[wnids[wnid_idx]])):

            # check whether missing
            if any(np.isclose([np.linalg.norm(wnid_observations[wnids[wnid_idx]][obs_idx]),
                                       np.linalg.norm(wnid_textf[wnids[wnid_idx]][obs_idx])], [0, 0])):
                if wnids[wnid_idx] not in missing:
                    missing[wnids[wnid_idx]] = []
                missing[wnids[wnid_idx]].append(obs_idx)

            # if we've already flagged this as a duplicate, no need to check
            if wnids[wnid_idx] in duplicates and obs_idx in duplicates[wnids[wnid_idx]]:
                continue

            # normalize for dot product later
            curr_dup = False
            imgf = wnid_observations[wnids[wnid_idx]][obs_idx]
            textf = wnid_textf[wnids[wnid_idx]][obs_idx]
            imgf_n = np.asarray(imgf)
            if np.linalg.norm(imgf) > 0:
                imgf_n /= np.linalg.norm(imgf)
            textf_n = np.asarray(textf)
            if np.linalg.norm(textf) > 0:
                textf_n /= np.linalg.norm(textf)

            # perform dot product between target vector and matrix
            imgf_p = np.inner(imgf_n, imgf_m)
            textf_p = np.inner(textf_n, textf_m)

            # inspect product to detect near-identical feature vectors
            for idx in range(0, len(row_wnid_obs_keys)):
                if row_wnid_obs_keys[idx][0] == wnid_idx and row_wnid_obs_keys[idx][1] == obs_idx:
                    continue
                if ((all(np.isclose([imgf_p[idx], np.linalg.norm(imgf), np.linalg.norm(imgf_m[idx])], [0, 0, 0])) or
                     np.isclose([imgf_p[idx]], [1])[0]) and
                    (all(np.isclose([textf_p[idx], np.linalg.norm(textf), np.linalg.norm(textf_m[idx])], [0, 0, 0])) or
                     np.isclose([textf_p[idx]], [1])[0])):

                    if wnids[row_wnid_obs_keys[idx][0]] not in duplicates:
                        duplicates[wnids[row_wnid_obs_keys[idx][0]]] = []
                    duplicates[wnids[row_wnid_obs_keys[idx][0]]].append(row_wnid_obs_keys[idx][1])
                    curr_dup = True

            # if duplicates were found, flag self
            if curr_dup and wnids[wnid_idx] not in known_duplicates:
                if wnids[wnid_idx] not in duplicates:
                    duplicates[wnids[wnid_idx]] = []
                duplicates[wnids[wnid_idx]].append(obs_idx)
    print "... done"

    # write duplicate map to file
    print "writing duplicate map to file..."
    with open(FLAGS_outfile, 'wb') as f:
        d = (duplicates, missing)
        pickle.dump(d, f)
    print "... done"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--urls_infile', type=str, required=True,
                        help="wnid graph used when getting observations")
    parser.add_argument('--obs_infile', type=str, required=True,
                        help="wnid observations file")
    parser.add_argument('--textf_infile', type=str, required=True,
                        help="wnid text observations file")
    parser.add_argument('--distributed', type=int, required=True,
                        help="1 if distributed, 0 otherwise")
    parser.add_argument('--target_wnid_idx', type=int, required=True,
                        help="the wnid idx for which to detect duplicate observations")
    parser.add_argument('--wnid_idx_start', type=int, required=True,
                        help="beginning of idx range to check")
    parser.add_argument('--wnid_idx_end', type=int, required=True,
                        help="end of idx range to check (exclusive)")
    parser.add_argument('--known_duplicates', type=str, required=True,
                        help="previously calculated, within-wnid duplicate map")
    parser.add_argument('--outfile', type=str, required=True,
                        help="output duplicate map")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    try:
        main()
    except Exception, e:
        print "Error: " + str(e)
