#!/usr/bin/env python
__author__ = 'jesse'
''' give this train and test np obs, a reconstruction attempt, and a wnid map for that attempt

    trains a multi-class classifier from the reconstruction attempt and wnid map to determine a wnid given
    an image
    tests this classifier's performance by iterating over all observations in test set and seeing whether
    it selects the correct wnid given an observation vector

'''

import argparse
import pickle
from sklearn import linear_model
import numpy


def main():

    # read infiles
    print "reading in observations, reconstruction, and wnid map..."
    f = open(FLAGS_wnid_obs_infile, 'rb')
    wnid_observations = pickle.load(f)
    f.close()
    f = open(FLAGS_np_test_infile, 'rb')
    test_observations = pickle.load(f)  # indexes into wnid_observations
    f.close()
    f = open(FLAGS_reconstruction_infile, 'rb')
    re_synsets, re_syn_obs = pickle.load(f)  # re_syn_obs keys by (np_idx, syn_idx) and indexes into wnid_obs
    f.close()
    f = open(FLAGS_reconstruction_wnids, 'rb')
    wnids = pickle.load(f)
    f.close()
    print "... done"

    # train a multi-class classifier to take in an observation and produce a wnid
    print "assembling data and training multi-class obs->wnid classifier..."
    data = []  # observation vectors
    labels = []  # wnid_idxs associated
    for wnid_idx in range(0, len(wnids)):
        for np_idx in re_synsets[wnid_idx]:
            for entry in re_syn_obs[(np_idx, wnid_idx)]:
                data.append(wnid_observations[entry[0]][entry[1]])
                labels.append(wnid_idx)
    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(data, labels)
    print "... done"

    # evaluate classifier against each test observation and report accuracy
    print "evaluating classifier against test observations..."
    num_correct = 0
    num_tested = 0
    for np in test_observations:
        for wnid, obs_idx in test_observations[np]:
            obs = wnid_observations[wnid][obs_idx]
            data = numpy.reshape(obs, (1, -1))
            pred_wnid = wnids[logreg.predict(data)]
            if pred_wnid == wnid:
                num_correct += 1
            num_tested += 1
    acc = float(num_correct) / num_tested
    print "... done; accuracy "+str(float(num_correct))+" / "+str(num_tested)+" = "+str(acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wnid_obs_infile', type=str, required=True,
                        help="set of wnid->observation vectors")
    parser.add_argument('--np_test_infile', type=str, required=True,
                        help="testing set of np observations not seen by algorithms so far")
    parser.add_argument('--reconstruction_infile', type=str, required=True,
                        help="reconstruction structures to be tested")
    parser.add_argument('--reconstruction_wnids', type=str, required=True,
                        help="reconstruction wnids aligned by scoring script")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
