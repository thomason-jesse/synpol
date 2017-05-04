#!/usr/bin/env python
__author__ = 'jesse'
''' give this train and test np obs and a reconstruction attempt

    trains a binary classifier for each synset and applies these to each image in the test set
    reports precision and recall at the np level (e.g. if `bat' is np, get it right if any synset
    containing `bat' returns True)

    negative examples for classification are drawn at random to match the size of positive classifications
    to save time

'''

import argparse
import pickle
import os
import time


def main():

    # read infiles
    print "reading in graph and reconstruction..."
    f = open(FLAGS_graph_infile, 'rb')
    wnids, synsets, nps, polysems = pickle.load(f)
    f.close()
    f = open(FLAGS_np_test_infile, 'rb')
    test_observations = pickle.load(f)  # indexes into wnid_observations
    f.close()
    f = open(FLAGS_reconstruction_infile, 'rb')
    re_synsets, re_syn_obs = pickle.load(f)  # re_syn_obs keys by (np_idx, syn_idx) and indexes into wnid_obs
    f.close()
    print "... done"

    # train a series of binary classifiers to determine whether an input image belongs to a re_synset
    print "launching jobs to train individual re_synset->True/False classifiers..."
    dec_pickle_fns = []
    for syn_idx in range(0, len(re_synsets)):
        out_fn = FLAGS_outfile + ".test_perf." + str(syn_idx) + ".pickle"
        cmd = ("condorify_gpu_email python train_and_run_np_classifier.py " +
               "--graph_infile " + FLAGS_graph_infile + " " +
               "--obs_urls " + FLAGS_obs_urls + " " +
               "--distributed " + str(FLAGS_distributed) + " " +
               "--wnid_obs_infile " + FLAGS_wnid_obs_infile + " " +
               "--np_test_infile " + FLAGS_np_test_infile + " " +
               "--reconstruction_infile "+FLAGS_reconstruction_infile + " " +
               "--outfile " + out_fn + " " +
               "--syn_idx " + str(syn_idx) + " " +
               out_fn + ".log")
        os.system(cmd)
        dec_pickle_fns.append(out_fn)
    print "... done"

    # poll for classifier decisions
    print "polling for finished classifier decisions..."
    decisions = [None for _ in range(0, len(re_synsets))]
    unfinished = range(0, len(re_synsets))
    while len(unfinished) > 0:
        time.sleep(10)
        newly_finished = []
        for syn_idx in unfinished:
            try:
                f = open(dec_pickle_fns[syn_idx], 'rb')
                classifier_decs = pickle.load(f)
                f.close()
            except:  # pickle hasn't been written all the way yet
                continue
            decisions[syn_idx] = classifier_decs
            newly_finished.append(syn_idx)
            os.system("rm "+dec_pickle_fns[syn_idx])
            os.system("rm "+dec_pickle_fns[syn_idx]+".log")
            os.system("rm err."+dec_pickle_fns[syn_idx].replace("/", "-")+".log")
        unfinished = [syn_idx for syn_idx in unfinished if syn_idx not in newly_finished]
        if len(newly_finished) > 0:
            print "... processed "+str(len(newly_finished))+" finished jobs " + \
                "("+str(len(re_synsets)-len(unfinished))+"/"+str(len(re_synsets))+")"
    print "... done"

    # run all classifiers on all images and record incremental progress
    print "measuring performance given decisions..."
    nps_precision = []
    nps_recall = []
    for np in range(0, len(nps)):
        if np not in test_observations:
            nps_precision.append(None)
            nps_recall.append(None)
            continue
        cm = [[0, 0], [0, 0]]  # indexed gold, eval
        for wnid, obs_idx in test_observations[np]:
            gold_nps = synsets[wnids.index(wnid)]
            eval_nps = []
            for syn_idx in range(0, len(re_synsets)):
                if (wnid, obs_idx) in decisions[syn_idx][np]:  # missing decisions not counted
                    dec = decisions[syn_idx][np][(wnid, obs_idx)]
                    if dec:
                        eval_nps.extend([np_idx for np_idx in re_synsets[syn_idx]
                                         if np_idx not in eval_nps])
            for np_idx in range(0, len(nps)):
                cm[np_idx in gold_nps][np_idx in eval_nps] += 1
        precision = float(cm[1][1]) / (cm[1][1]+cm[0][1]) if (cm[1][1]+cm[0][1]) > 0 else None
        recall = float(cm[1][1]) / (cm[1][1]+cm[1][0]) if (cm[1][1]+cm[1][0]) > 0 else None
        nps_precision.append(precision)
        nps_recall.append(recall)
    print "... done"
    # for np in test_observations:  # DEBUG
    #     print nps[np], nps_precision[np], nps_recall[np]  # DEBUG

    # report results
    print "calculating precision and recall globally and writing out results..."
    nps_precision_non_none = [entry for entry in nps_precision if entry is not None]
    nps_recall_non_none = [entry for entry in nps_recall if entry is not None]
    avg_precision = sum(nps_precision_non_none) / float(len(nps_precision_non_none)) \
        if len(nps_precision_non_none) > 0 else None
    avg_recall = sum(nps_recall_non_none) / float(len(nps_recall_non_none)) \
        if len(nps_recall_non_none) > 0 else None
    f = open(FLAGS_outfile, 'wb')
    d = [nps_precision, nps_recall]
    pickle.dump(d, f)
    f.close()
    f1 = 2*(avg_precision*avg_recall)/(avg_precision+avg_recall) \
        if avg_precision is not None and avg_recall is not None else None
    f = open(FLAGS_perf_outfile, 'w')
    d = [avg_precision, avg_recall, f1]
    pickle.dump(d, f)
    f.close()
    print "... done; avg_precision="+str(avg_precision)+", avg_recall="+str(avg_recall)+", avg f1="+str(f1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_infile', type=str, required=True,
                        help="graph used when gathering observations")
    parser.add_argument('--obs_urls', type=str, required=True,
                        help="wnids -> urls")
    parser.add_argument('--wnid_obs_infile', type=str, required=True,
                        help="set of wnid->observation vectors")
    parser.add_argument('--distributed', type=int, required=True,
                        help="whether wnid->obs vectors are distributed on disk")
    parser.add_argument('--np_test_infile', type=str, required=True,
                        help="testing set of np observations not seen by algorithms so far")
    parser.add_argument('--reconstruction_infile', type=str, required=True,
                        help="reconstruction structures to be tested")
    parser.add_argument('--outfile', type=str, required=True,
                        help="location to pickle precision and recall over nps")
    parser.add_argument('--perf_outfile', type=str, required=True,
                        help="location to pickle precision, recall, f1 list")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
