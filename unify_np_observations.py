#!/usr/bin/env python
__author__ = 'jesse'
''' give this sets of np_observations

    outputs all those together

'''

import argparse
import pickle


def main():

    # read infiles
    print "reading in and unifying observations..."
    np_observations = {}
    for fn in FLAGS_np_obs_infiles.split(','):
        f = open(fn, 'rb')
        obs = pickle.load(f)
        f.close()
        for key in obs:
            if key not in np_observations:
                np_observations[key] = []
            np_observations[key].extend(obs[key])
    print "... done"

    # write np observations to file
    print "writing unified np observations to file..."
    f = open(FLAGS_outfile, 'wb')
    d = np_observations
    pickle.dump(d, f)
    f.close()
    print "... done"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--np_obs_infiles', type=str, required=True,
                        help="np observations files comma separated")
    parser.add_argument('--outfile', type=str, required=True,
                        help="np observation outfile")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
