#!/usr/bin/env python
__author__ = 'jesse'
''' takes a wnid graph, a set of wnid -> text observation maps, and a serialized lsi model and
    calculates the textual features for the wnid observations given the observed text and outputs
    a map from wnid -> text features

'''

import argparse
import pickle
import os
import time


def main():

    # read infiles
    print "reading in urls, observations, lsi model, and dictionary..."
    f = open(FLAGS_wnid_urls, 'rb')
    wnid_urls = pickle.load(f)
    wnids = wnid_urls.keys()
    f.close()
    print "... read graph"
    if FLAGS_text_obs_unified > 0:
        f = open(FLAGS_text_obs_infile, 'rb')
        wnid_text = pickle.load(f)
        f.close()
        print "... read text observations"
    print "... done"

    # calculate lsi textual features from text corpus observations
    print "launching jobs to calculate lsi textual features from text observations..."
    remaining_wnid_jobs = []
    for wnid_idx in range(0, len(wnids)):
        wnid = wnids[wnid_idx]
        launch_job = False
        if FLAGS_text_obs_unified > 0:
            wnid_text_obs = FLAGS_text_obs_infile
            if wnid in wnid_text and len(wnid_text[wnid]) > 0:
                launch_job = True
        else:
            wnid_text_obs = str(wnid_idx) + "_" + FLAGS_text_obs_infile
            try:
                with open(wnid_text_obs, 'rb') as pf:
                    _ = pickle.load(pf)
                    launch_job = True
            except (IOError, EOFError):
                print "... WARNING: missing pickle for wnid " + str(wnid_idx) + "; cannot get features for it"
        if launch_job:
            outf = str(wnid_idx) + "_lsi_temp.pickle" if FLAGS_text_obs_unified > 0 else str(wnid_idx) + "_" + FLAGS_outfile
            cmd = ("condorify_gpu_email python get_textf_from_lsi_for_wnid.py " +
                   "--target_wnid " + wnid + " " +
                   "--text_obs_infile " + wnid_text_obs + " " +
                   "--lsi_dictionary " + FLAGS_lsi_dictionary + " " +
                   "--lsi_dictionary " + FLAGS_lsi_dictionary + " " +
                   "--tfidf_model " + FLAGS_tfidf_model + " " +
                   "--lsi_model " + FLAGS_lsi_model + " " +
                   "--lsi_fsize " + str(FLAGS_lsi_fsize) + " " +
                   "--outfile " + outf +
                   str(wnid_idx) + "_lsi_temp")
            os.system(cmd)
            remaining_wnid_jobs.append(wnid_idx)
    print "... done"

    # poll for jobs finished and build merged duplicates structure
    if FLAGS_text_obs_unified > 0:
        print "merging textf results into map as they become available..."
        wnid_textf = {}
        while len(remaining_wnid_jobs) > 0:
            time.sleep(10)  # poll for finished scripts every 10 seconds
            newly_finished_jobs = []
            for wnid_idx in remaining_wnid_jobs:
                log_fn = str(wnid_idx) + "_lsi_temp"
                lsi_fn = log_fn + ".pickle"
                if os.path.isfile(lsi_fn):
                    try:
                        with open(lsi_fn, 'rb') as pf:
                            new_textf = pickle.load(pf)
                    except (IOError, EOFError, ValueError, KeyError):
                        continue
                    newly_finished_jobs.append(wnid_idx)
                    os.system("rm " + lsi_fn)
                    os.system("rm err." + log_fn)
                    os.system("rm " + log_fn)
                    wnid_textf[wnids[wnid_idx]] = new_textf
            remaining_wnid_jobs = [wnid_idx for wnid_idx in remaining_wnid_jobs if wnid_idx not in newly_finished_jobs]
            if len(newly_finished_jobs) > 0:
                print ("... " + str(len(remaining_wnid_jobs)) + " wnids remain after adding wnids: " +
                       str(newly_finished_jobs))
            whether_to_continue = raw_input("continue checks(Y/n)? ")  # handle this weird shit
            if whether_to_continue == 'n':
                break

        # write textf
        print "writing wnid -> textf observations to file..."
        with open(FLAGS_outfile, 'wb') as f:
            d = wnid_textf
            pickle.dump(d, f)
        print "... done"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wnid_urls', type=str, required=True,
                        help="wnid urls used when getting text observations")
    parser.add_argument('--text_obs_infile', type=str, required=True,
                        help="wnid text observations file")
    parser.add_argument('--text_obs_unified', type=int, required=True,
                        help="whether wnid text observations are in one file or one per wnid (1 for one file)")
    parser.add_argument('--lsi_dictionary', type=str, required=True,
                        help="dictionary of words used in lsi model")
    parser.add_argument('--tfidf_model', type=str, required=True,
                        help="tfidf model used by lsi")
    parser.add_argument('--lsi_model', type=str, required=True,
                        help="serialized lsi model")
    parser.add_argument('--lsi_fsize', type=int, required=True,
                        help="number of features in lsi")
    parser.add_argument('--outfile', type=str, required=True,
                        help="output text features from w2v")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
