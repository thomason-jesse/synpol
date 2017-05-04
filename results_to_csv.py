#!/usr/bin/env python
__author__ = 'jesse'
''' pass this a directory in which to look for class and graph results files

    outputs a csv of result statistics
'''

import argparse
import os
import pickle


def main():

    results_dir = FLAGS_results_dir
    out_fn = FLAGS_outfile

    # establish headers
    headers = ["pol_syn", "fold", "graph_p", "graph_r", "graph_f1", "class_p", "class_r", "class_f1"]

    # trace dir and gather data
    d = {}  # indexed by (pol, syn, fold) triple; value is graph then class entries
    for root, dirs, files in os.walk(results_dir):
        for fn in files:
            fn_dot_parts = fn.split('.')
            if len(fn_dot_parts) == 3:
                fn_name_parts = fn_dot_parts[0].split('_')
                pol = fn_name_parts[1]
                syn = fn_name_parts[2]
                fold = fn_name_parts[3]
                key = (pol+"_"+syn, fold)
                if fn_dot_parts[1] == 'graph':
                    f = open(os.path.join(root, fn), 'rb')
                    r = pickle.load(f)
                    f.close()
                    if key not in d:
                        d[key] = r
                    else:
                        r.extend(d[key])
                        d[key] = r
                elif fn_dot_parts[1] == 'class':
                    f = open(os.path.join(root, fn), 'rb')
                    r = pickle.load(f)
                    f.close()
                    if key not in d:
                        d[key] = r
                    else:
                        d[key].extend(r)

    # write data to csv
    f = open(out_fn, 'w')
    f.write(','.join(headers)+'\n')
    for key in d:
        if len(d[key]) == 6:
            f.write(','.join(key)+','+','.join([str(n) for n in d[key]])+'\n')
        else:
            print "WARNING: missing some stats for key "+str(key)  # DEBUG
    f.close()

    # check for completion
    pols = ["gap", "dpgmm", "ms"]
    syns = ["none", "gap", "dpgmm", "ms"]
    folds = ["0", "1", "2", "3", "4"]
    for pol in pols:
        for syn in syns:
            for fold in folds:
                key = (pol+"_"+syn, fold)
                if key not in d:
                    print "WARNING: missing all stats for key "+str(key)  # DEBUG


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True,
                        help="directory where results files live")
    parser.add_argument('--outfile', type=str, required=True,
                        help="csv to write to")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
