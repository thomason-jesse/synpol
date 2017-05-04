#!/usr/bin/env python
__author__ = 'jesse'
''' takes a wnid graph, set of wnid -> observation maps, and a set of wnid -> textf observation maps,
    and creates and outputs a new set of wnid -> observation / textf / url maps that contain only observations
    that are unique (e.g. not duplicates in data) and have both text and visual information available

'''

import argparse
import numpy as np
import pickle
import os
import time


def main():

    distributed = True if FLAGS_distributed == 1 else False

    # read infiles
    print "reading input files..."
    if not distributed:
        f = open(FLAGS_obs_infile, 'rb')
        wnid_observations = pickle.load(f)
        f.close()
        print "... read image observations"
        f = open(FLAGS_textf_infile, 'rb')
        wnid_textf = pickle.load(f)
        f.close()
        print "... read text features"
    f = open(FLAGS_url_infile, 'rb')
    wnid_urls = pickle.load(f)
    wnids = wnid_urls.keys()
    f.close()
    print "... read urls"
    try:
        f = open(FLAGS_known_within_wnid_duplicates, 'rb')
        duplicates = pickle.load(f)
        f.close()
        print "... read known within wnid duplicates"
    except (IOError, EOFError):
        print "WARNING: no known within wnid duplicates file loaded"
        duplicates = {}
    try:
        f = open(FLAGS_known_across_wnid_duplicates, 'rb')
        across_wnid_dups = pickle.load(f)
        for wnid in across_wnid_dups:
            if wnid not in duplicates:
                duplicates[wnid] = across_wnid_dups
            else:
                for dup in across_wnid_dups[wnid]:
                    if dup not in duplicates[wnid]:
                        duplicates[wnid].append(dup)
        across_wnids = across_wnid_dups.keys()
        f.close()
        print "... read known across wnid duplicates"
    except (IOError, EOFError):
        print "WARNING: no known across wnid duplicates file loaded"
        across_wnids = []
    print "... done"

    # launch jobs to detect duplicates based on feature matches
    print "launching jobs to detect duplicate observations..."
    num_obs = 0
    remaining_wnid_jobs = []
    for wnid_idx in range(0, len(wnids)):
        if not distributed and wnids[wnid_idx] not in wnid_observations:
            continue
        if wnids[wnid_idx] in across_wnids:
            continue
        if distributed:
            try:
                with open(str(wnid_idx) + "_" + FLAGS_obs_infile) as f:
                    wnid_observations = pickle.load(f)
                with open(str(wnid_idx) + "_" + FLAGS_textf_infile) as f:
                    wnid_textf = {wnids[wnid_idx]: pickle.load(f)}
            except (IOError, EOFError):
                print "... WARNING: no pickle for idx " + str(wnid_idx)  # DEBUG
                continue
        num_obs += len(wnid_observations[wnids[wnid_idx]])
        cmd = ("condorify_gpu_email python find_duplicate_observations_in_range.py " +
               "--urls_infile " + FLAGS_url_infile + " " +
               "--obs_infile " + FLAGS_obs_infile + " " +
               "--textf_infile " + FLAGS_textf_infile + " " +
               "--distributed " + str(FLAGS_distributed) + " " +
               "--target_wnid_idx " + str(wnid_idx) + " " +
               "--wnid_idx_start " + str(wnid_idx+1) + " " +  # doesn't count self; include known missing/dups
               "--wnid_idx_end " + str(len(wnids)) + " " +
               "--known_duplicates " + FLAGS_known_duplicates + " " +
               "--outfile " + '-'.join([str(wnid_idx), str(wnid_idx), str(len(wnids))]) + "_duplicate_temp.pickle " +
               '-'.join([str(wnid_idx), str(wnid_idx), str(len(wnids))]) + "_duplicate_temp")
        os.system(cmd)
        remaining_wnid_jobs.append(wnid_idx)
    print "... done; detecting over " + str(num_obs) + " observations across " + str(len(remaining_wnid_jobs)) + " wnids"

    # poll for jobs finished and build merged duplicates structure
    print "merging duplicate map and missing results as they become available..."
    missing = {}
    num_dups = 0
    num_miss = 0
    while len(remaining_wnid_jobs) > 0:
        time.sleep(10)  # poll for finished scripts every 10 seconds
        newly_finished_jobs = []
        for wnid_idx in remaining_wnid_jobs:
            log_fn = '-'.join([str(wnid_idx), str(wnid_idx), str(len(wnids))]) + "_duplicate_temp"
            dup_fn = log_fn + ".pickle"
            if os.path.isfile(dup_fn):
                try:
                    pf = open(dup_fn, 'rb')
                    new_duplicates, new_missing = pickle.load(pf)
                    pf.close()
                except (IOError, EOFError):
                    continue
                newly_finished_jobs.append(wnid_idx)
                os.system("rm " + dup_fn)
                os.system("rm err." + log_fn)
                os.system("rm " + log_fn)
                for wnid_jdx in new_duplicates:
                    if wnid_jdx not in duplicates:
                        duplicates[wnid_jdx] = new_duplicates[wnid_jdx]
                        num_dups += len(new_duplicates[wnid_jdx])
                    else:
                        for obs_idx in new_duplicates[wnid_jdx]:
                            if obs_idx not in duplicates[wnid_jdx]:
                                duplicates[wnid_jdx].append(obs_idx)
                                num_dups += 1
                for wnid_jdx in new_missing:
                    if wnid_jdx not in missing:
                        missing[wnid_jdx] = new_missing[wnid_jdx]
                        num_miss += len(new_missing[wnid_jdx])
                    else:
                        for obs_idx in new_missing[wnid_jdx]:
                            if obs_idx not in missing[wnid_jdx]:
                                missing[wnid_jdx].append(obs_idx)
                                num_miss += 1
        remaining_wnid_jobs = [wnid_idx for wnid_idx in remaining_wnid_jobs if wnid_idx not in newly_finished_jobs]
        if len(newly_finished_jobs) > 0:
            print ("... " + str(len(remaining_wnid_jobs)) + " wnids remain after merging in wnids: " +
                   str(newly_finished_jobs))
        whether_to_continue = raw_input("... continue checks(Y/n)? ")  # handle this weird shit
        if whether_to_continue == 'n':
            break
    print "... done; detected " + str(num_dups) + " duplicates and " + str(num_miss) + " with missing feature vectors"

    if not distributed:

        # build new observation maps removing duplicates and missing features
        print "building new observation maps..."
        new_imgf = {}
        new_textf = {}
        new_urls = {}
        for wnid in wnids:
            if wnid in wnid_observations:
                for obs_idx in range(0, len(wnid_observations[wnid])):
                    if ((wnid in duplicates and obs_idx in duplicates[wnid]) or
                            (wnid in missing and obs_idx in missing[wnid])):
                        if wnid not in new_imgf:
                            new_imgf[wnid] = []
                            new_textf[wnid] = []
                            new_urls[wnid] = []
                        new_imgf[wnid].append(wnid_observations[wnid][obs_idx])
                        new_textf[wnid].append(wnid_textf[wnid][obs_idx])
                        new_urls[wnid].append(wnid_urls[wnid][obs_idx])
        print "... done"

        # output finished maps
        print "writing new image features to file..."
        with open(FLAGS_obs_outfile, 'wb') as f:
            d = new_imgf
            pickle.dump(d, f)
        print "... done"
        print "writing new text features to file..."
        with open(FLAGS_textf_outfile, 'wb') as f:
            d = new_textf
            pickle.dump(d, f)
        print "... done"
        print "writing new observation urls to file..."
        with open(FLAGS_url_outfile, 'wb') as f:
            d = new_urls
            pickle.dump(d, f)
        print "... done"

    else:

        # write missing and duplicates to file
        print "writing missing or duplicates to file"
        missing_or_duplicate = duplicates
        for wnid in missing:
            if wnid not in missing_or_duplicate:
                missing_or_duplicate[wnid] = missing[wnid]
            else:
                for obs_idx in missing[wnid]:
                    if obs_idx not in missing_or_duplicate[wnid]:
                        missing_or_duplicate[wnid].append(obs_idx)
        with open(FLAGS_missing_and_duplicate_list, 'wb') as f:
            d = missing_or_duplicate
            pickle.dump(d, f)
        print "... done"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--obs_infile', type=str, required=True,
                        help="wnid observations file")
    parser.add_argument('--textf_infile', type=str, required=True,
                        help="wnid text observations file")
    parser.add_argument('--url_infile', type=str, required=True,
                        help="wnid observation urls file")
    parser.add_argument('--distributed', type=int, required=True,
                        help="1 if distributed, 0 otherwise")
    parser.add_argument('--obs_outfile', type=str, required=True,
                        help="wnid image observations out; unused if distributed")
    parser.add_argument('--textf_outfile', type=str, required=True,
                        help="wnid text observations out; unused if distributed")
    parser.add_argument('--url_outfile', type=str, required=True,
                        help="wnid observation urls out; unused if distributed")
    parser.add_argument('--missing_and_duplicate_list', type=str, required=True,
                        help="pickle of wnids that have missing features or are duplicates; unused if not distributed")
    parser.add_argument('--known_within_wnid_duplicates', type=str, required=True,
                        help="previously calculated, within-wnid duplicate map")
    parser.add_argument('--known_across_wnid_duplicates', type=str, required=True,
                        help="previously calculated, across-wnid duplicate map")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
