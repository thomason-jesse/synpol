#!/usr/bin/env python
__author__ = 'jesse'

import argparse
import copy
from kl_functions import multivariate_kl_distance
import numpy as np
import pickle
import sys


def main():

    # read infiles
    print "reading in means, vars, and conn matrix..."
    with open(FLAGS_means_infile, 'rb') as f:
        means = pickle.load(f)
    with open(FLAGS_vars_infile, 'rb') as f:
        vars = pickle.load(f)
        orig_vars = copy.copy(vars)
    with open(FLAGS_conn_infile, 'rb') as f:
        conn = pickle.load(f)
    print "... done"

    # move through connectivity looking for bogus distances and correcting them
    print "updating connectivity matrix to calculate degenerate distances"
    d = 0
    f_avgs = [None for _ in range(len(vars[0]))]
    for syn_idx in range(len(vars)):
        for syn_jdx in range(syn_idx+1, len(vars)):
            if conn[syn_idx][syn_jdx] == float(sys.maxint):

                # replace zeros in relevant variance vector features with average variance in those features
                print "... looking for zeros in (" + str(syn_idx) + ", " + str(syn_jdx) + ")"
                for s_idx in [syn_idx, syn_jdx]:
                    for f_idx in range(len(vars[s_idx])):
                        if len(np.nonzero(vars[s_idx][f_idx])[0]) == 0:

                            # load pre-calculated average or calculate average if we haven't encountered yet
                            if f_avgs[f_idx] is None:
                                f_vars = [orig_vars[idx][f_idx] for idx in range(len(orig_vars))
                                          if len(np.nonzero(vars[idx][f_idx])[0]) == 1]
                                if len(f_vars) == 0:
                                    print ("...... WARNING: whole feature " + str(s_idx) + ": " +
                                           str(f_idx) + " variances are zero; setting var to 1.0")
                                    avg = 1.0
                                else:
                                    avg = sum(f_vars) / len(f_vars)
                                f_avgs[f_idx] = avg
                            vars[s_idx][f_idx] = f_avgs[f_idx]
                            print ("...... changed zero feature " + str(s_idx) + ": " + str(f_idx) +
                                   " to average " + str(f_avgs[f_idx]))

                # re-perform distance calculation
                dist = multivariate_kl_distance(means[syn_idx], vars[syn_idx],
                                                means[syn_jdx], vars[syn_jdx])
                conn[syn_idx][syn_jdx] = dist
                d += 1
                print "... calculated new distance d(" + str(syn_idx) + ", " + str(syn_jdx) + ") = " + str(dist)
    print "... done; updated " + str(d) + " distances"

    # write updated conn matrix to file
    print "writing stats to file..."
    with open(FLAGS_conn_outfile, 'wb') as f:
        pickle.dump(conn, f)
    print "... done"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--means_infile', type=str, required=True,
                        help="mean vectors")
    parser.add_argument('--vars_infile', type=str, required=True,
                        help="variance vectors")
    parser.add_argument('--conn_infile', type=str, required=True,
                        help="conn matrix")
    parser.add_argument('--conn_outfile', type=str, required=True,
                        help="conn matrix with previously degenerate distances updated")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
