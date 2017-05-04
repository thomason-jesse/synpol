#!/usr/bin/env python
__author__ = 'jesse'
''' pass this a synpol data graph pickle

    outputs statistics about the data graph as well as some hyperparameters of interest
'''

import argparse
import pickle
import math


def main():

    # read in synpol data graph structures
    print "reading synpol data graph pickle and np observations..."
    f = open(FLAGS_graph_infile, 'rb')
    wnids, synsets, noun_phrases, polysems = pickle.load(f)
    f.close()
    f = open(FLAGS_np_obs_infile, 'rb')
    np_observations = pickle.load(f)
    f.close()
    f = open(FLAGS_wnid_obs_urls, 'rb')
    wnid_urls = pickle.load(f)
    f.close()
    print "... done"

    # initial stats
    print "synsets max: "+str(max([len(nps) for nps in synsets]))
    print "synsets avg: "+str(sum([len(nps) for nps in synsets]) / float(len(synsets)))
    print "synsets min: "+str(min([len(nps) for nps in synsets]))
    print "polysemy max: "+str(max([len(polysems[np]) for np in polysems]))
    print "polysemy avg: "+str(sum([len(polysems[np]) for np in polysems]) / float(len(polysems.keys())))
    print "polysemy min: "+str(min([len(polysems[np]) for np in polysems]))

    # observation stats
    size_wnid_obs = [len(wnid_urls[wnid]) for wnid in wnids if wnid in wnid_urls]
    size_mean = sum(size_wnid_obs) / float(len(size_wnid_obs))
    print "synset obs max: " + str(max(size_wnid_obs))
    print "synset obs avg: " + str(size_mean)
    print "synset obs min: " + str(min(size_wnid_obs))
    print "synset obs stddev: " + str(math.sqrt(
        sum([(size-size_mean)**2 for size in size_wnid_obs]) / len(size_wnid_obs)))

    # count
    print "calculating stats of interest over noun phrases..."
    num_syn = 0
    num_pol = 0
    num_both = 0
    num_neither = 0
    for np_idx in range(0, len(noun_phrases)):
        syn = False
        pol = np_idx in polysems
        for syn_idx in range(0, len(synsets)):
            if np_idx in synsets[syn_idx] and len(synsets[syn_idx]) > 1:
                syn = True
                break
        if syn and not pol:
            num_syn += 1
        elif pol and not syn:
            num_pol += 1
        elif syn and pol:
            num_both += 1
        else:
            num_neither += 1
    print "... done; num syn only: "+str(num_syn)+", num pol only: "+str(num_pol) + \
        ", num both: "+str(num_both)+", num neither: "+str(num_neither)

    # calculate word senses per cluster
    # corresponds to number of noun phrases in each synset over total num synsets
    num_senses = sum([len(syn) for syn in synsets])
    word_senses_per_cluster = num_senses / float(len(synsets))
    print "word senses per cluster: "+str(word_senses_per_cluster)

    # calculate `alpha' density parameter for DPGMM
    # alpha = sum_np { |C_np| / ln(|N_np|) } / |NP| for C_np the number of clusters (synsets) to which np belongs,
    # N_np the number of total observations for np, and |NP| the total number of noun phrases
    # this is an average estimate for alpha over the whole development set
    alpha = 0.0
    nps_considered = 0
    for np_idx in range(0, len(noun_phrases)):
        if np_idx in np_observations:
            c = sum([1 if np_idx in syn else 0 for syn in synsets])
            n = len(np_observations[np_idx])
            if n > 1:
                alpha += c / math.log(n)
            nps_considered += 1
    alpha /= nps_considered
    print "alpha_pol density parameter: "+str(alpha)

    # calculate `alpha' density parameter for DPGMM operating over sense means to form synsets
    # then alpha = |C| / ln(|N|) for C the total number of synsets, |N| the number of senses
    alpha_syn = len(synsets) / math.log(num_senses)
    print "alpha_syn density parameter: "+str(alpha_syn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_infile', type=str, required=True,
                        help="input synpol data graph pickle")
    parser.add_argument('--wnid_obs_urls', type=str, required=True,
                        help="wnid obs urls for the given graph")
    parser.add_argument('--np_obs_infile', type=str, required=True,
                        help="np observations for the given graph")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
