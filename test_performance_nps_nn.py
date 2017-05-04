#!/usr/bin/env python
__author__ = 'jesse'
''' give this train and test np obs and a reconstruction attempt

    trains a 1-nearest-neighbor classifier from the average feature vector of each synset
    returns average precision/recall calculated for each np, where test images with the np in their
    tags are counted correct if they attach to a synset that contains that np

'''

import argparse
from gap_statistic_functions import reevaluate_centers
import numpy
import pickle
import os
from reconstruct_graph_from_np_observations import DiskDictionary
import time


def main():

    distributed = True if FLAGS_distributed == 1 else False
    use_condor = True if FLAGS_use_condor == 1 else False

    # read infiles
    print "reading in graph and reconstruction..."
    f = open(FLAGS_graph_infile, 'rb')
    wnids, synsets, nps, polysems = pickle.load(f)
    f.close()
    f = open(FLAGS_obs_urls, 'rb')
    wnid_urls = pickle.load(f)
    f.close()
    if not distributed:
        f = open(FLAGS_wnid_obs_infile, 'rb')
        wnid_imgfs = pickle.load(f)
        f.close()
        f = open(FLAGS_wnid_textf_infile, 'rb')
        wnid_textfs = pickle.load(f)
        f.close()
    else:
        max_floats = 125000000 * 1
        wnid_imgfs = DiskDictionary(FLAGS_wnid_obs_infile, max_floats, wnid_urls.keys())
        wnid_textfs = DiskDictionary(FLAGS_wnid_textf_infile, max_floats, wnid_urls.keys())
    f = open(FLAGS_np_test_infile, 'rb')
    test_observations = pickle.load(f)  # indexes into wnid_observations
    f.close()
    f = open(FLAGS_reconstruction_infile, 'rb')
    re_synsets, re_syn_obs = pickle.load(f)  # re_syn_obs keys by (np_idx, syn_idx) and indexes into wnid_obs
    f.close()
    print "... done"

    # calculate the redundancy of feature types for early fusion based on input weighting
    print "... calculating redundancy coefficients..."
    imgf_weight = FLAGS_proportion_imgf_versus_textf
    imgf_fv_size = len(wnid_imgfs[wnids[0]][0])
    textf_fv_size = len(wnid_textfs[wnids[0]][0])
    if imgf_weight == 0:
        imgf_red = 0
        textf_red = 1
    elif imgf_weight == 1:
        imgf_red = 1
        textf_red = 0
    elif imgf_fv_size > textf_fv_size:
        imgf_red = 1
        textf_red = int(((imgf_fv_size * (1 - imgf_weight)) / (imgf_weight * textf_fv_size)) + 0.5)
    elif imgf_fv_size < textf_fv_size:
        imgf_red = int(((imgf_weight * textf_fv_size) / (imgf_fv_size * (1 - imgf_weight))) + 0.5)
        textf_red = 1
    else:
        imgf_red = textf_red = 1
    print ("...done; to achieve an imgf weight of " + str(imgf_weight) + ", calculated imgf redundancy " +
           str(imgf_red) + " (" + str(imgf_fv_size) + " features) with textf redundancy " + str(textf_red) +
           " (" + str(textf_fv_size) + " features)")

    # calculate synset means
    print "calculating re_synset means..."
    means = []
    clusters = {}
    for re_idx in range(0, len(re_synsets)):
        obs = []
        for np_idx in re_synsets[re_idx]:
            if imgf_red > 0:
                new_obs = [wnid_imgfs[entry[0]][entry[1]]
                           for entry in re_syn_obs[(np_idx, re_idx)]]
            else:
                new_obs = [wnid_textfs[entry[0]][entry[1]]
                           for entry in re_syn_obs[(np_idx, re_idx)]]
            for _ in range(1, imgf_red):
                for entry_idx in range(0, len(re_syn_obs[(np_idx, re_idx)])):
                    entry = re_syn_obs[(np_idx, re_idx)][entry_idx]
                    new_obs[entry_idx].extend(wnid_imgfs[entry[0]][entry[1]])
            for _ in range(1, textf_red):
                for entry_idx in range(0, len(re_syn_obs[(np_idx, re_idx)])):
                    entry = re_syn_obs[(np_idx, re_idx)][entry_idx]
                    new_obs[entry_idx].extend(wnid_textfs[entry[0]][entry[1]])
            obs.extend(new_obs)
        n_obs = numpy.asarray(obs)
        clusters[re_idx] = n_obs
    means.extend(reevaluate_centers(clusters, len(clusters[0][0])))
    print "... done; got " + str(len(means)) + " means for " + str(len(re_synsets)) + " reconstructed synsets"

    # launch job to build classifier
    print "writing means and launching job to build 1-nearest-neighbor classifier from means..."
    means_fn = FLAGS_outfile + ".means"
    classifier_fn = FLAGS_outfile + ".classifier"
    os.system("rm " + means_fn)
    os.system("rm " + classifier_fn)
    with open(means_fn, 'wb') as f:
        pickle.dump(means, f)
    cmd = ("python train_one_nn_classifier.py " +
           "--means " + means_fn + " " +
           "--outfile " + classifier_fn)
    if use_condor:
        cmd = "condorify_gpu_email_largemem " + cmd + " " + FLAGS_outfile + ".nn"
    os.system(cmd)
    print "... done"

    # poll for finished classifier before continuing
    print "polling for finished classifier..."
    classifier_trained = False
    while not classifier_trained:
        time.sleep(60)
        try:
            with open(classifier_fn, 'rb') as f:
                _ = pickle.load(f)
            f.close()
            classifier_trained = True
        except (IOError, EOFError, KeyError, ValueError):  # pickle hasn't been written all the way yet
            continue
    print "... done"

    # launch jobs to load classifier and run on each test image
    print "launching jobs for each np in test observations to get nearest neighbors"
    unfinished = []
    for np_idx in test_observations:
        if len(test_observations[np_idx]) > 0:
            out_fn = FLAGS_outfile + "_" + str(np_idx) + ".one_nn"
            cmd = ("python get_one_nn_of_observations.py " +
                   "--classifier " + classifier_fn + " " +
                   "--obs_urls " + FLAGS_obs_urls + " " +
                   "--wnid_obs_infile " + FLAGS_wnid_obs_infile + " " +
                   "--wnid_textf_infile " + FLAGS_wnid_textf_infile + " " +
                   "--imgf_red " + str(imgf_red) + " " +
                   "--textf_red " + str(textf_red) + " " +
                   "--distributed " + str(FLAGS_distributed) + " " +
                   "--np_test_infile " + FLAGS_np_test_infile + " " +
                   "--np_idx " + str(np_idx) + " " +
                   "--outfile " + out_fn)
            if use_condor:
                cmd = "condorify_gpu_email_largemem " + cmd + " " + out_fn + ".log"
            os.system(cmd)
            unfinished.append(np_idx)
    print "... done; launched " + str(len(unfinished)) + " jobs for the test_observation nps"

    # reduce to populate decisions structure
    print "polling for finished jobs and reducing results to decisions structure..."
    decisions = []
    for re_idx in range(0, len(re_syn_obs)):
        decisions.append({})
    while len(unfinished) > 0:
        newly_finished = []
        for np_idx in unfinished:
            out_fn = FLAGS_outfile + "_" + str(np_idx) + ".one_nn"
            try:
                f = open(out_fn, 'rb')
                classifier_decs = pickle.load(f)
                f.close()
            except (IOError, EOFError, KeyError, ValueError):  # pickle hasn't been written all the way yet
                continue
            for test_idx in range(0, len(classifier_decs)):
                re_idx = classifier_decs[test_idx]
                if np_idx not in decisions[re_idx]:
                    decisions[re_idx][np_idx] = []
                decisions[re_idx][np_idx].append(test_observations[np_idx][test_idx])
            os.system("rm " + out_fn)
            if use_condor:
                os.system("rm " + out_fn + ".log")
                os.system("rm err." + out_fn.replace("/", "-") + ".log")
            newly_finished.append(np_idx)
        unfinished = [np_idx for np_idx in unfinished if np_idx not in newly_finished]
        if len(newly_finished) > 0:
            print "... processed "+str(len(newly_finished))+" finished jobs " + \
                "("+str(len(test_observations)-len(unfinished))+"/"+str(len(test_observations))+")"
    print "... done"

    # remove means and classifier files
    os.system("rm " + means_fn)
    os.system("rm " + classifier_fn)

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
            for re_idx in range(0, len(re_synsets)):
                if np in decisions[re_idx] and (wnid, obs_idx) in decisions[re_idx][np]:
                    eval_nps.extend([np_idx for np_idx in re_synsets[re_idx]
                                     if np_idx not in eval_nps])
            for np_idx in range(0, len(nps)):
                cm[np_idx in gold_nps][np_idx in eval_nps] += 1
        precision = float(cm[1][1]) / (cm[1][1]+cm[0][1]) if (cm[1][1]+cm[0][1]) > 0 else None
        recall = float(cm[1][1]) / (cm[1][1]+cm[1][0]) if (cm[1][1]+cm[1][0]) > 0 else None
        nps_precision.append(precision)
        nps_recall.append(recall)
    print "... done"

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
    parser.add_argument('--wnid_textf_infile', type=str, required=True,
                        help="set of wnid->text feature vectors")
    parser.add_argument('--proportion_imgf_versus_textf', type=float, required=True,
                        help="proportion in [0, 1] of weight given to image versus text features")
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
    parser.add_argument('--use_condor', type=int, required=True,
                        help="whether to use condor")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
