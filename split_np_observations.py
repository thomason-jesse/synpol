#!/usr/bin/env python
__author__ = 'jesse'
''' give this a set of np_observations

    outputs two np_observations structures of specified size and name partitioning the original

'''

import argparse
import pickle
import random
import sys


def main():

    # read infiles
    print "reading in observations..."
    f = open(FLAGS_np_obs_infile, 'rb')
    np_observations = pickle.load(f)
    f.close()
    print "... done"

    partitions = [float(s) for s in FLAGS_partitions.split(',')]
    if sum(partitions) != 1:
        sys.exit("ERROR: partitions must sum to 1")
    outfiles = FLAGS_outfiles.split(',')
    if len(partitions) != len(outfiles):
        sys.exit("ERROR: num partitions must match num outfiles")

    partitioned_obs = []
    for idx in range(0, len(partitions)):
        partitioned_obs.append({})
    for np in np_observations:
        obs = np_observations[np]
        if len(obs) < len(partitions):
            print "WARNING: np '"+str(np)+"' has fewer than "+str(len(partitions)) + \
                  " observations ("+str(len(obs))+") which will be discarded"
            for idx in range(0, len(partitions)):
                partitioned_obs[idx][np] = []
            continue
        random.shuffle(obs)
        for idx in range(0, len(partitions)):
            partitioned_obs[idx][np] = [obs[idx]]
        oidx = len(partitions)
        for idx in range(0, len(partitions)):
            while sum([len(partitioned_obs[jdx][np]) for jdx in range(0, len(partitions))]) < len(obs) \
                    and float(len(partitioned_obs[idx][np]))/len(obs) < partitions[idx]:
                partitioned_obs[idx][np].append(obs[oidx])
                oidx += 1

    # write synsets, syn_obs of induced topology
    print "writing partitioned observations to file..."
    for idx in range(0, len(outfiles)):
        f = open(outfiles[idx], 'wb')
        d = partitioned_obs[idx]
        pickle.dump(d, f)
        f.close()
    print "... done"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--np_obs_infile', type=str, required=True,
                        help="np observations file")
    parser.add_argument('--partitions', type=str, required=True,
                        help="amount of data to give each partition; should sum to 1")
    parser.add_argument('--outfiles', type=str, required=True,
                        help="comma separated outfiles for partitions")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
