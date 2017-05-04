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
import sys
import time


def main():

    _CONDOR_MAX_JOBS = 1000

    # read infiles
    print "reading in graph, observations, and reconstruction..."
    f = open(FLAGS_graph_infile, 'rb')
    wnids, synsets, nps, polysems = pickle.load(f)
    f.close()
    f = open(FLAGS_wnid_obs_url_infile, 'rb')
    wnid_urls = pickle.load(f)
    f.close()
    f = open(FLAGS_reconstruction_infile, 'rb')
    _ = pickle.load(f)  # just ensure exists
    f.close()
    f = open(FLAGS_np_train_obs, 'rb')
    np_train_observations = pickle.load(f)
    f.close()
    train_observations = []
    for np_idx in range(0, len(nps)):
        if np_idx in np_train_observations:
            train_observations.extend(np_train_observations[np_idx])
    print "... done"

    # cut synsets down to size by observing whether they actually have observations (urls)
    old_wnids = copy.deepcopy(wnids)
    old_synsets = copy.deepcopy(synsets)
    synsets = [old_synsets[wnid_idx]
               for wnid_idx in range(0, len(old_wnids))
               if old_wnids[wnid_idx] in wnid_urls]

    print "alternating launching and gathering jobs to calculate paired matches"
    syn_pairs = pair_matches = 0
    re_pairs = None
    unlaunched_jobs = range(0, len(synsets))
    remaining_jobs = []
    while len(unlaunched_jobs) > 0 or len(remaining_jobs) > 0:
        newly_launched = []
        for syn_idx in unlaunched_jobs:
            if len(remaining_jobs) >= _CONDOR_MAX_JOBS:
                break
            cmd = ("python score_reconstruction_pairedf_sub.py " +
                   "--graph_infile " + FLAGS_graph_infile + " " +
                   "--wnid_obs_url_infile " + FLAGS_wnid_obs_url_infile + " " +
                   "--reconstruction_infile " + FLAGS_reconstruction_infile + " " +
                   "--np_train_obs " + str(FLAGS_np_train_obs) + " " +
                   "--syn_idx " + str(syn_idx) + " " +
                   "--outfile " + FLAGS_perf_outfile + "_" + str(syn_idx) + ".sub.pickle ")
            cmd = "condorify_gpu_email " + cmd + FLAGS_perf_outfile + "_" + str(syn_idx) + ".sub"
            os.system(cmd)
            newly_launched.append(syn_idx)
            remaining_jobs.append(syn_idx)
        unlaunched_jobs = [job for job in unlaunched_jobs if job not in newly_launched]
        if len(newly_launched) > 0:
            print "...... " + str(len(unlaunched_jobs)) + " jobs remain after launching " + str(len(newly_launched))

        newly_finished = []
        for syn_idx in remaining_jobs:
            log_fn = FLAGS_perf_outfile + "_" + str(syn_idx) + ".sub"
            row_fn = log_fn + ".pickle"
            if os.path.isfile(row_fn):
                try:
                    pf = open(row_fn, 'rb')
                    pairs, new_re_pairs, matches = pickle.load(pf)
                    pf.close()
                except (IOError, EOFError, ValueError):
                    continue
                newly_finished.append(syn_idx)
                syn_pairs += pairs
                pair_matches += matches
                if re_pairs is None:
                    re_pairs = new_re_pairs
                if re_pairs != new_re_pairs:
                    print ("WARNING: mismatched re_synsets pairs calculations;"
                           "old " + str(re_pairs) + " vs new " + str(new_re_pairs))
                os.system("rm " + log_fn)
                os.system("rm err." + log_fn.replace('/', '-'))
                os.system("rm " + row_fn)
        remaining_jobs = [job for job in remaining_jobs if job not in newly_finished]
        if len(newly_finished) > 0:
            print "...... " + str(len(remaining_jobs)) + " jobs remain after gathering " + str(len(newly_finished))

        if len(remaining_jobs) >= _CONDOR_MAX_JOBS or len(newly_launched) == 0:
            time.sleep(60)
    print ("... done; got syn_pairs=" + str(syn_pairs) + ", re_pairs=" + str(re_pairs) +
           ", pair_matches=" + str(pair_matches))

    print "calculating p, r, and f from pairings"
    p = pair_matches / float(re_pairs)
    r = pair_matches / float(syn_pairs)
    fm = (2 * p * r) / (p + r)
    print "... done; got p=" + str(p) + ", r=" + str(r) + ", fm=" + str(fm)

    # write synsets, syn_obs of induced topology
    print "writing stats to file..."
    with open(FLAGS_perf_outfile, 'w') as f:
        d = [p, r, fm]
        pickle.dump(d, f)
    print "... done"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_infile', type=str, required=True,
                        help="wnid graph used to construct observations")
    parser.add_argument('--wnid_obs_url_infile', type=str, required=True,
                        help="wnid observations url file (faster to load; don't actually need numbers)")
    parser.add_argument('--reconstruction_infile', type=str, required=True,
                        help="reconstruction structures to be tested")
    parser.add_argument('--np_train_obs', type=str, required=True,
                        help="observations to consider")
    parser.add_argument('--perf_outfile', type=str, required=True,
                        help="write paired precision, recall, f-measure list here")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
