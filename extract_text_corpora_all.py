#!/usr/bin/env python
__author__ = 'jesse'
''' give this a set of np_observations

    outputs synsets and observation pairing attempting to reconstruct original wnid_graph based on observations alone

'''

import argparse
import pickle
import os
import time


def main():

    # read infiles
    print "reading in observation urls..."
    f = open(FLAGS_wnid_obs_urls_infile, 'rb')
    wnid_observations_urls = pickle.load(f)
    f.close()
    print "... done"

    # split wnids and write smaller input pickles and launch jobs
    wnids = wnid_observations_urls.keys()
    print "launching jobs to gather texts..."
    out_pickle_urls = []
    for wnid_idx in range(0, len(wnids)):
        outf = str(wnid_idx) + "_" + FLAGS_outfile
        try:
            with open(outf, 'rb') as pf:
                _ = pickle.load(pf)
                print "... skipping launch of written " + outf
        except (IOError, EOFError):  # pickle hasn't been written previously 
            small_urls = {wnids[wnid_idx]: wnid_observations_urls[wnids[wnid_idx]]}
            obsf = FLAGS_wnid_obs_urls_infile + "_" + str(wnid_idx)
            with open(obsf, 'wb') as f:
                pickle.dump(small_urls, f)
            cmd = ("condorify_gpu_email python extract_text_corpora.py " +
                   "--wnid_obs_urls_infile " + obsf + " " +
                   "--outfile " + outf + " " +
                   str(wnid_idx)+"_extract_text_corpora.log")
            os.system(cmd)
            out_pickle_urls.append(outf)
            print "... launched for " + outf
    print "... done"

    # gather results of jobs
    print "monitoring job results..."
    # wnid_observations_texts = {}
    while len(out_pickle_urls) > 0:
        time.sleep(60*5)  # poll for finished scripts every 5 minutes
        newly_finished = []
        for outf in out_pickle_urls:
            if os.path.isfile(outf):
                try:
                    with open(outf, 'rb') as pf:
                        _ = pickle.load(pf)
                    newly_finished.append(outf)
                except (IOError, EOFError):  # pickle hasn't been written all the way yet
                    continue
                # for wnid in small_observation_texts:
                #     wnid_observations_texts[wnid] = small_observation_texts[wnid]
                # print "...... got texts for " + str(len(small_observation_texts.keys())) + " wnids from " + outf
                # os.system("rm " + outf)
        out_pickle_urls = [fn for fn in out_pickle_urls
                           if fn not in newly_finished]
        if len(newly_finished) > 0:
            print ("... got " + str(len(newly_finished)) + " new jobs (" + str(len(wnids)-len(out_pickle_urls))
                   + "/" + str(len(wnids)) + ")")
    print "... done"

    # write finished observations
    # print "writing all wnids with extracted text to file..."
    # with open(FLAGS_outfile, 'wb') as f:
    #     d = wnid_observations_texts
    #     pickle.dump(d, f)
    # print "... done"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wnid_obs_urls_infile', type=str, required=True,
                        help="wnid observation urls file")
    parser.add_argument('--outfile', type=str, required=True,
                        help="output pickled map from synsets to instance text lists")
    parser.add_argument('--partial_texts', type=str, required=False,
                        help="existing texts for some subset of wnids")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
