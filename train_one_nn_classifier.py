#!/usr/bin/env python
__author__ = 'jesse'
''' give this a set of observation means

    outputs trained 1-nn classifier

'''

import argparse
import pickle
import numpy
from sklearn.neighbors import KNeighborsClassifier


def main():

    # read in obs
    f = open(FLAGS_means, 'rb')
    means = pickle.load(f)
    f.close()

    # do the grunt work
    np_means = numpy.asarray(means)
    one_nn = KNeighborsClassifier(n_neighbors=1)
    one_nn.fit(np_means, range(0, len(np_means)))  # each neighbor is a single synset class

    # write num_k, n_obs_classes to pickle
    f = open(FLAGS_outfile, 'wb')
    d = one_nn
    pickle.dump(d, f)
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--means', type=str, required=True,
                        help="observations to operate over")
    parser.add_argument('--outfile', type=str, required=True,
                        help="output pickled return values from function")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
