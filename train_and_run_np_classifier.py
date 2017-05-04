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
from sklearn import svm
import numpy
import random
from reconstruct_graph_from_np_observations import DiskDictionary


def main():

    distributed = True if FLAGS_distributed == 1 else False

    # read infiles
    print "reading in graph, observations, and reconstruction..."
    f = open(FLAGS_graph_infile, 'rb')
    wnids, synsets, nps, polysems = pickle.load(f)
    f.close()
    f = open(FLAGS_obs_urls, 'rb')
    wnid_urls = pickle.load(f)
    f.close()
    if not distributed:
        f = open(FLAGS_wnid_obs_infile, 'rb')
        wnid_observations = pickle.load(f)
        f.close()
    else:
        max_floats = 125000000 * 1
        wnid_observations = DiskDictionary(FLAGS_wnid_obs_infile, max_floats, wnid_urls.keys())
    f = open(FLAGS_np_test_infile, 'rb')
    test_observations = pickle.load(f)  # indexes into wnid_observations
    f.close()
    f = open(FLAGS_reconstruction_infile, 'rb')
    re_synsets, re_syn_obs = pickle.load(f)  # re_syn_obs keys by (np_idx, syn_idx) and indexes into wnid_obs
    f.close()
    print "... done"

    # train a series of binary classifiers to determine whether an input image belongs to a re_synset
    print "assembling data and training individual re_synset->True/False classifier..."
    classifier = None
    syn_idx = FLAGS_syn_idx
    data = []  # observation vectors
    labels = []  # bool associated
    for np_idx in re_synsets[syn_idx]:
        key = (np_idx, syn_idx)
        for entry in re_syn_obs[key]:
            data.append(wnid_observations[entry[0]][entry[1]])
            labels.append(True)
    if True in labels:
        used_entries = {}  # triple of syn_jdx, syn_np_jdx, syn_np_entry_jdx
        num_true_labels = len(labels)
        num_false_labels = 0
        # sample false observations up to the number of true observations
        while num_false_labels < num_true_labels:
            syn_jdx = random.randint(0, len(re_synsets)-1)
            if syn_idx == syn_jdx or len(re_synsets[syn_jdx]) == 0:
                continue
            syn_np_jdx = random.randint(0, len(re_synsets[syn_jdx])-1)
            np_jdx = re_synsets[syn_jdx][syn_np_jdx]
            if len(re_syn_obs[(np_jdx, syn_jdx)]) == 0:
                continue
            syn_np_entry_jdx = random.randint(0, len(re_syn_obs[(np_jdx, syn_jdx)])-1)
            if (syn_jdx in used_entries and syn_np_jdx in used_entries[syn_jdx] and
                    syn_np_entry_jdx in used_entries[syn_jdx][syn_np_jdx]):
                continue
            entry = re_syn_obs[(np_jdx, syn_jdx)][syn_np_entry_jdx]
            data.append(wnid_observations[entry[0]][entry[1]])
            labels.append(False)
            num_false_labels += 1
            if syn_jdx not in used_entries:
                used_entries[syn_jdx] = {}
            if syn_np_jdx not in used_entries[syn_jdx]:
                used_entries[syn_jdx][syn_np_jdx] = []
            used_entries[syn_jdx][syn_np_jdx].append(syn_np_entry_jdx)
        if False in labels:
            lin_svm = svm.LinearSVC()
            lin_svm.fit(data, labels)
            classifier = lin_svm
    print "... done"

    # run all classifiers on all images and record incremental progress
    print "running synset classifier against all test images and recording decisions..."
    classifier_decs = {}
    for np in range(0, len(nps)):
        classifier_decs[np] = {}
        if np not in test_observations:
            continue
        for wnid, obs_idx in test_observations[np]:
            obs = wnid_observations[wnid][obs_idx]
            data = numpy.reshape(obs, (1, -1))
            dec = False
            if classifier is not None:
                dec = classifier.predict(data)
            classifier_decs[np][(wnid, obs_idx)] = dec
    print "... done"

    # report results
    print "writing decisions to file..."
    f = open(FLAGS_outfile, 'wb')
    d = classifier_decs
    pickle.dump(d, f)
    f.close()


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
    parser.add_argument('--syn_idx', type=int, required=True,
                        help="individual syn_idx to build classifier and evaluate on")
    parser.add_argument('--outfile', type=str, required=True,
                        help="location to pickle precision and recall over nps")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
