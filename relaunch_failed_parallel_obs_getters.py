#!/usr/bin/env python
__author__ = 'jesse'
''' pass this a synpol data graph pickle

    outputs maps from wnids to observation feature vectors
    that will need to be unified later
'''

import argparse
import pickle
import os


def main():

    # read in synpol data graph structures
    print "reading synpol data graph pickle"
    f = open(FLAGS_data_infile, 'rb')
    wnids, synsets, noun_phrases, polysems = pickle.load(f)
    f.close()
    print "... done"

    # split pickle into temporary ones for job to be done
    num_wnids_per_job = float(len(wnids)) / FLAGS_jobs_to_launch
    head_wnid = 0
    print "launching jobs..."
    for jdx in range(0, FLAGS_jobs_to_launch):
        tail_wnid = head_wnid+num_wnids_per_job
        if tail_wnid > len(wnids) or jdx == FLAGS_jobs_to_launch-1:
            tail_wnid = len(wnids)

        # launch a condor job if pickle doesn't exist
        obs_fn = FLAGS_outfile+"."+str(int(head_wnid))+"-"+str(int(tail_wnid))
        if not os.path.isfile(obs_fn):
            fn = "condor_temp.pickle."+str(int(head_wnid))+"-"+str(int(tail_wnid))
            url_fn = FLAGS_outfile+".urls."+str(int(head_wnid))+"-"+str(int(tail_wnid))
            cmd = "condorify_gpu_email python get_observation_features_for_wnids.py " + \
                  "--data_infile "+fn+" --obs_outfile "+obs_fn+" "+" --url_outfile "+url_fn+" " + \
                  "--observations_per_np "+str(FLAGS_observations_per_np)+" " + \
                  str(int(head_wnid))+"-"+str(int(tail_wnid))+".log"
            print "running: '"+cmd+"'"
            os.system(cmd)

        head_wnid = tail_wnid

    print "... done"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_infile', type=str, required=True,
                        help="input synpol data graph pickle")
    parser.add_argument('--jobs_to_launch', type=int, required=True,
                        help="number of condor jobs to use for the task")
    parser.add_argument('--observations_per_np', type=int, required=True,
                        help="max number of observations to gather per noun phrase in each wnid")
    parser.add_argument('--outfile', type=str, required=True,
                        help="output pickles of wnid->observation features maps")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
