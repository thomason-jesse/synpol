#!/usr/bin/env python
__author__ = 'jesse'
''' unify observation pickles and clean up data pickles

'''

import argparse
import pickle
import os


def main():

    # read infile
    f = open(FLAGS_infile, 'rb')
    wnids, _, _, _ = pickle.load(f)
    f.close()

    # get temporary pickle names and deal with them each
    num_wnids_per_job = float(len(wnids)) / FLAGS_jobs_launched
    head_wnid = 0
    wnid_observations = {}
    wnid_urls = {}
    print "unifying over jobs..."
    for jdx in range(0, FLAGS_jobs_launched):
        tail_wnid = head_wnid+num_wnids_per_job
        if tail_wnid > len(wnids) or jdx == FLAGS_jobs_launched-1:
            tail_wnid = len(wnids)

        # unpickle observations and add to globals
        obs_fn = FLAGS_outfile+"."+str(int(head_wnid))+"-"+str(int(tail_wnid))
        url_fn = FLAGS_outfile+".urls."+str(int(head_wnid))+"-"+str(int(tail_wnid))
        f = open(obs_fn, 'rb')
        obs = pickle.load(f)
        f.close()
        f = open(url_fn, 'rb')
        url = pickle.load(f)
        f.close()
        for wnid in obs:
            if wnid in wnid_observations:
                print "WARNING: wnid '"+wnid+"' being overwritten by subsequent pickle"
            wnid_observations[wnid] = obs[wnid]
        for wnid in url:
            if wnid in wnid_urls:
                print "WARNING: wnid '"+wnid+"' url being overwritten by subsequent pickle"
            wnid_urls[wnid] = url[wnid]
        print "... got observations for "+str(len(wnid_observations.keys()))+" synsets"

        head_wnid = tail_wnid
    print "... done"

    # output final observations
    print "writing all observations to file..."
    f = open(FLAGS_outfile, 'wb')
    pickle.dump(wnid_observations, f)
    f.close()
    f = open(FLAGS_outfile+".urls", 'wb')
    pickle.dump(wnid_urls, f)
    f.close()
    print "... done"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, required=True,
                        help="wnid graph used when getting partial observations")
    parser.add_argument('--jobs_launched', type=int, required=True,
                        help="number of condor jobs to use for the task")
    parser.add_argument('--outfile', type=str, required=True,
                        help="output pickles of wnid->observation features maps")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
