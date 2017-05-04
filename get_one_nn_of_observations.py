#!/usr/bin/env python
__author__ = 'jesse'
''' give this train and test np obs and a reconstruction attempt

    trains a 1-nearest-neighbor classifier from the average feature vector of each synset
    returns average precision/recall calculated for each np, where test images with the np in their
    tags are counted correct if they attach to a synset that contains that np

'''

import argparse
import numpy
import pickle
from reconstruct_graph_from_np_observations import DiskDictionary


def main():

    imgf_red = FLAGS_imgf_red
    textf_red = FLAGS_textf_red
    np_idx = FLAGS_np_idx
    distributed = True if FLAGS_distributed == 1 else False

    # read infiles
    print "reading in observations and trained classifier..."
    f = open(FLAGS_np_test_infile, 'rb')
    test_observations = pickle.load(f)  # indexes into wnid_observations
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
    f = open(FLAGS_classifier, 'rb')
    one_nn = pickle.load(f)
    f.close()
    print "... done"

    print "calculating feature representations of test observations..."
    if imgf_red > 0:
        obs = [wnid_imgfs[entry[0]][entry[1]]
               for entry in test_observations[np_idx]]
    else:
        obs = [wnid_textfs[entry[0]][entry[1]]
               for entry in test_observations[np_idx]]
    for _ in range(1, imgf_red):
        for entry_idx in range(0, len(test_observations[np_idx])):
            entry = test_observations[np_idx][entry_idx]
            obs[entry_idx].extend(wnid_imgfs[entry[0]][entry[1]])
    for _ in range(1, textf_red):
        for entry_idx in range(0, len(test_observations[np_idx])):
            entry = test_observations[np_idx][entry_idx]
            obs[entry_idx].extend(wnid_textfs[entry[0]][entry[1]])
    obs_n = numpy.asarray(obs)
    print "... done"

    print "running one nearest-neighbor classifier on test observations..."
    d = one_nn.predict(obs_n)
    print ".. done"

    # write num_k, n_obs_classes to pickle
    print "writing class decisions to file..."
    f = open(FLAGS_outfile, 'wb')
    pickle.dump(d, f)
    f.close()
    print "... done"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--classifier', type=str, required=True,
                        help="trained one-nn classifier")
    parser.add_argument('--obs_urls', type=str, required=True,
                        help="wnids -> urls")
    parser.add_argument('--wnid_obs_infile', type=str, required=True,
                        help="set of wnid->observation vectors")
    parser.add_argument('--wnid_textf_infile', type=str, required=True,
                        help="set of wnid->text feature vectors")
    parser.add_argument('--imgf_red', type=int, required=True,
                        help="image feature redundancy")
    parser.add_argument('--textf_red', type=int, required=True,
                        help="text feature redundancy")
    parser.add_argument('--distributed', type=int, required=True,
                        help="whether wnid->obs vectors are distributed on disk")
    parser.add_argument('--np_test_infile', type=str, required=True,
                        help="testing set of np observations not seen by algorithms so far")
    parser.add_argument('--np_idx', type=int, required=True,
                        help="the noun phrase idx of interest")
    parser.add_argument('--outfile', type=str, required=True,
                        help="location to pickle classifier decisions array")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
