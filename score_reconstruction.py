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
    re_synsets, re_syn_obs = pickle.load(f)
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

    print "calculating homogeneity, completeness, and v-measure..."
    print "... calculating a_ij matrix of class i present in cluster j"
    a = []  # indexed first by syn_idx then by re_idx
    n = 0

    unlaunched_jobs = range(0, len(synsets))
    remaining_jobs = []
    print "...... alternating launching/polling jobs to calculate overlap rows"
    while len(unlaunched_jobs) > 0 or len(remaining_jobs) > 0:
        newly_launched = []
        for syn_idx in unlaunched_jobs:
            if len(remaining_jobs) >= _CONDOR_MAX_JOBS:
                break
            cmd = ("python score_reconstruction_sub.py " +
                   "--graph_infile " + FLAGS_graph_infile + " " +
                   "--wnid_obs_url_infile " + FLAGS_wnid_obs_url_infile + " " +
                   "--reconstruction_infile " + FLAGS_reconstruction_infile + " " +
                   "--np_train_obs " + str(FLAGS_np_train_obs) + " " +
                   "--syn_idx " + str(syn_idx) + " " +
                   "--outfile " + FLAGS_perf_outfile + "_" + str(syn_idx) + ".sub.pickle ")
            cmd = "condorify_gpu_email " + cmd + FLAGS_perf_outfile + "_" + str(syn_idx) + ".sub"
            os.system(cmd)
            a.append([])
            newly_launched.append(syn_idx)
            remaining_jobs.append(syn_idx)
        unlaunched_jobs = [job for job in unlaunched_jobs if job not in newly_launched]
        if len(newly_launched) > 0:
            print "......... " + str(len(unlaunched_jobs)) + " jobs remain after launching " + str(len(newly_launched))

        newly_finished = []
        for syn_idx in remaining_jobs:
            log_fn = FLAGS_perf_outfile + "_" + str(syn_idx) + ".sub"
            row_fn = log_fn + ".pickle"
            if os.path.isfile(row_fn):
                try:
                    pf = open(row_fn, 'rb')
                    row, add_n = pickle.load(pf)
                    pf.close()
                except (IOError, EOFError, ValueError):
                    continue
                newly_finished.append(syn_idx)
                n += add_n
                a[syn_idx] = row
                os.system("rm " + log_fn)
                os.system("rm err." + log_fn.replace('/', '-'))
                os.system("rm " + row_fn)
        remaining_jobs = [job for job in remaining_jobs if job not in newly_finished]
        if len(newly_finished) > 0:
            print "......... " + str(len(remaining_jobs)) + " jobs remain after gathering " + str(len(newly_finished))

        if len(remaining_jobs) >= _CONDOR_MAX_JOBS or len(newly_launched) == 0:
            time.sleep(60)

    print "...... done; n=" + str(n)
    print "... calculating H(S) for S the set of gold standard classes"
    h_s = 0
    for syn_idx in range(0, len(synsets)):
        sum_ai = sum([a[syn_idx][re_idx] for re_idx in range(0, len(re_synsets))])
        avg_overlap = sum_ai / float(n)
        if avg_overlap > 0:
            h_s -= avg_overlap * math.log(avg_overlap)
    print "...... H(S)= "+str(h_s)

    print "... calculating H(S|K) for S the gold standard classes and K the reconstructed classes"
    f = open(FLAGS_perf_outfile+ "_an", 'wb')
    d = [a, n]
    pickle.dump(d, f)
    f.close()

    print "...... launching/polling jobs to calculate H(S|K) contribution from each K"
    h_s_k = 0
    unlaunched_jobs = range(0, len(re_synsets))
    remaining_jobs = []
    while len(unlaunched_jobs) > 0 or len(remaining_jobs) > 0:
        newly_launched = []
        for re_idx in unlaunched_jobs:
            if len(remaining_jobs) >= _CONDOR_MAX_JOBS:
                break
            cmd = ("python score_reconstruction_sk.py " +
                   "--graph_infile " + FLAGS_graph_infile + " " +
                   "--wnid_obs_url_infile " + FLAGS_wnid_obs_url_infile + " " +
                   "--member_matrix " + FLAGS_perf_outfile+"_an" + " " +
                   "--re_idx " + str(re_idx) + " " +
                   "--outfile " + FLAGS_perf_outfile + "_" + str(re_idx) + ".sk.pickle ")
            cmd = "condorify_gpu_email " + cmd + FLAGS_perf_outfile + "_" + str(re_idx) + ".sk"
            os.system(cmd)
            newly_launched.append(re_idx)
            remaining_jobs.append(re_idx)
        unlaunched_jobs = [job for job in unlaunched_jobs if job not in newly_launched]
        if len(newly_launched) > 0:
            print "......... " + str(len(unlaunched_jobs)) + " jobs remain after launching " + str(len(newly_launched))

        newly_finished = []
        for re_idx in remaining_jobs:
            log_fn = FLAGS_perf_outfile + "_" + str(re_idx) + ".sk"
            row_fn = log_fn + ".pickle"
            if os.path.isfile(row_fn):
                try:
                    pf = open(row_fn, 'rb')
                    h_s_k_part = pickle.load(pf)
                    pf.close()
                except (IOError, EOFError):
                    continue
                newly_finished.append(re_idx)
                h_s_k -= h_s_k_part
                os.system("rm " + log_fn)
                os.system("rm err." + log_fn.replace('/', '-'))
                os.system("rm " + row_fn)
        remaining_jobs = [job for job in remaining_jobs if job not in newly_finished]
        if len(newly_finished) > 0:
            print "......... " + str(len(remaining_jobs)) + " jobs remain after gathering " + str(len(newly_finished))

        if len(remaining_jobs) >= _CONDOR_MAX_JOBS or len(newly_launched) == 0:
            time.sleep(60)

    print "...... H(S|K)= "+str(h_s_k)
    h = 1.0 - (h_s_k / h_s) if h_s != 0 else 1

    print "... calculating H(K) for K the set of reconstructed classes"
    h_k = 0
    for re_idx in range(0, len(re_synsets)):
        sum_aj = sum([a[syn_idx][re_idx] for syn_idx in range(0, len(synsets))])
        avg_overlap = sum_aj / float(n)
        if avg_overlap > 0:
            h_k -= avg_overlap * math.log(avg_overlap)
    print "...... H(K)= " + str(h_k)

    print "...... launching/polling jobs to calculate H(K|S) contribution from each S"
    h_k_s = 0
    unlaunched_jobs = range(0, len(synsets))
    remaining_jobs = []
    while len(unlaunched_jobs) > 0 or len(remaining_jobs) > 0:
        newly_launched = []
        for syn_idx in unlaunched_jobs:
            if len(remaining_jobs) >= _CONDOR_MAX_JOBS:
                break
            cmd = ("python score_reconstruction_ks.py " +
                   "--reconstruction_infile " + FLAGS_reconstruction_infile + " " +
                   "--member_matrix " + FLAGS_perf_outfile+"_an" + " " +
                   "--syn_idx " + str(syn_idx) + " " +
                   "--outfile " + FLAGS_perf_outfile + "_" + str(syn_idx) + ".ks.pickle ")
            cmd = "condorify_gpu_email " + cmd + FLAGS_perf_outfile + "_" + str(syn_idx) + ".ks"
            os.system(cmd)
            remaining_jobs.append(syn_idx)
            newly_launched.append(syn_idx)
        unlaunched_jobs = [job for job in unlaunched_jobs if job not in newly_launched]
        if len(newly_launched) > 0:
            print "......... " + str(len(unlaunched_jobs)) + " jobs remain after launching " + str(len(newly_launched))

        newly_finished = []
        for syn_idx in remaining_jobs:
            log_fn = FLAGS_perf_outfile + "_" + str(syn_idx) + ".ks"
            row_fn = log_fn + ".pickle"
            if os.path.isfile(row_fn):
                try:
                    pf = open(row_fn, 'rb')
                    h_k_s_part = pickle.load(pf)
                    pf.close()
                except (IOError, EOFError):
                    continue
                newly_finished.append(syn_idx)
                h_k_s -= h_k_s_part
                os.system("rm " + log_fn)
                os.system("rm err." + log_fn.replace('/', '-'))
                os.system("rm " + row_fn)
        remaining_jobs = [job for job in remaining_jobs if job not in newly_finished]
        if len(newly_finished) > 0:
            print "......... " + str(len(remaining_jobs)) + " jobs remain after gathering " + str(len(newly_finished))

        if len(remaining_jobs) >= _CONDOR_MAX_JOBS or len(newly_launched) == 0:
            time.sleep(60)

    print "...... H(K|S)= " + str(h_k_s)
    os.system("rm " + FLAGS_perf_outfile+"_an")

    c = 1.0 - (h_k_s / h_k) if h_k != 0 else 1
    # calculate V-measure
    v = (2 * h * c) / (h + c)
    print "... done; with H(S)=" + str(h_s) + ", H(S|K)=" + str(h_s_k) + ", H(K)=" + str(h_k) + ", H(K|S)=" + str(h_k_s)
    print "... got homogeneity=" + str(h) + ", completeness=" + str(c) + ", v-measure=" + str(v)

    # write synsets, syn_obs of induced topology
    print "writing reconstructed to gold wnid mapping and stats to file..."
    f = open(FLAGS_perf_outfile, 'w')
    d = [h, c, v]
    pickle.dump(d, f)
    f.close()
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
                        help="write homogeneity, completeness, and v-measure list here")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
