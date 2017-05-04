#!/usr/bin/env python
__author__ = 'jesse'

import argparse
import pickle
import sys


# Takes in an observation (wnid, obs_idx) and a syn_obs structure and returns the syn_idx
# in the syn_obs dictionary that contains the specified observation.
def get_syn_idx_of_observations(obs, syn_obs):
    r = None
    for np_idx, syn_idx in syn_obs:
        if obs in syn_obs[(np_idx, syn_idx)]:
            r = syn_idx
            break
    return r


# Takes in a pair of synsets and syn_obs structures.
# Returns an array parallel to synsets of maps, each of which is indexed by alt_synset idxs
# and contains a pair of arrays of observations matching each alt_synset idx
# of syn_obs idxs that appear in the same synset with differing underlying word senses
# while also appearing in the two distinct synsets in alt_synsets.
def get_mismatch_maps(synsets, syn_obs, alt_synsets, alt_syn_obs):

    # syn_obs is a dictionary keyed by (np_idx, syn_idx) for syn_idx keying into the synsets array.
    # synsets is an array containing arrays of np_idxs that are descriptor nps for the synset.
    # np_idxs index into the nps structure provided by the graph_infile.

    mismatch_maps = []
    pairs_found = 0
    mismatches = 0
    syn_not_in_alt = []
    for syn_idx in range(len(synsets)):  # Find all pairs in each syn_idx.
        mismatch_map = {}  # For this synset.
        if len(synsets[syn_idx]) > 1:
            alt_syn_idxs = {}  # From syn_obs entries to alt_synsets idx.
            for np_idx_pos in range(len(synsets[syn_idx])):
                np_idx = synsets[syn_idx][np_idx_pos]
                for obs_a in syn_obs[(np_idx, syn_idx)]:

                    # Find alt_syn_idx for obs_a.
                    if obs_a not in alt_syn_idxs:
                        alt_syn_idxs[obs_a] = get_syn_idx_of_observations(obs_a, alt_syn_obs)
                    if alt_syn_idxs[obs_a] is None:
                        if obs_a not in syn_not_in_alt:
                            syn_not_in_alt.append(obs_a)
                        continue

                    for np_jdx_pos in range(np_idx_pos+1, len(synsets[syn_idx])):
                        np_jdx = synsets[syn_idx][np_jdx_pos]
                        for obs_b in syn_obs[(np_jdx, syn_idx)]:

                            # Find alt_syn_jdx of obs_b.
                            if obs_b not in alt_syn_idxs:
                                alt_syn_idxs[obs_b] = get_syn_idx_of_observations(obs_b, alt_syn_obs)
                            if alt_syn_idxs[obs_b] is None:
                                if obs_b not in syn_not_in_alt:
                                    syn_not_in_alt.append(obs_b)
                                continue

                            # Add to mismatch_map
                            if alt_syn_idxs[obs_a] != alt_syn_idxs[obs_b]:
                                mismatch_a = (np_idx, alt_syn_idxs[obs_a])
                                mismatch_b = (np_jdx, alt_syn_idxs[obs_b])
                                if mismatch_a not in mismatch_map and mismatch_b not in mismatch_map:
                                    s = sorted([mismatch_a, mismatch_b])
                                    mismatch_map[s[0]] = {}
                                    if s[0] == mismatch_a:
                                        mismatch_map[s[0]][s[1]] = [(obs_a, obs_b)]
                                    else:
                                        mismatch_map[s[0]][s[1]] = [(obs_b, obs_a)]
                                    mismatches += 1
                                elif mismatch_a not in mismatch_map:
                                    if mismatch_a not in mismatch_map[mismatch_b]:
                                        mismatch_map[mismatch_b][mismatch_a] = []
                                        mismatches += 1
                                    mismatch_map[mismatch_b][mismatch_a].append((obs_b, obs_a))
                                else:
                                    if mismatch_b not in mismatch_map[mismatch_a]:
                                        mismatch_map[mismatch_a][mismatch_b] = []
                                        mismatches += 1
                                    mismatch_map[mismatch_a][mismatch_b].append((obs_a, obs_b))
                                pairs_found += 1

                                # (226, 249, 8368, 3796, 6646)  # DEBUG

        mismatch_maps.append(mismatch_map)

        if (syn_idx + 1) % 100 == 0:  # DEBUG
            print ("\t" + str(syn_idx + 1) + " / " + str(len(synsets)) + " synsets processed; " +
                   str(pairs_found) + " pairs found across " + str(mismatches) + " synset mismatches")  # DEBUG

    print str(len(syn_not_in_alt)) + " observations were found in syn but not in alt!"  # DEBUG

    return mismatch_maps


# Takes in a pair of synsets and syn_obs structures.
# Returns the set of observation pairs in syn_obs that come from the same synset but distinct
# underlying word senses while also appearing in distinct synsets in the alt structures.
def get_pairs(synsets, syn_obs, alt_synsets, alt_syn_obs):

    # syn_obs is a dictionary keyed by (np_idx, syn_idx) for syn_idx keying into the synsets array.
    # synsets is an array containing arrays of np_idxs that are descriptor nps for the synset.
    # np_idxs index into the nps structure provided by the graph_infile.

    pairs = []
    for syn_idx in range(len(synsets)):  # Find all pairs in each syn_idx.
        if len(synsets[syn_idx]) > 1:
            alt_syn_idxs = {}  # From syn_obs entries to alt_synsets idx.
            for np_idx_pos in range(len(synsets[syn_idx])):
                np_idx = synsets[syn_idx][np_idx_pos]
                for obs_a in syn_obs[(np_idx, syn_idx)]:

                    # Find alt_syn_idx for obs_a.
                    if obs_a not in alt_syn_idxs:
                        alt_syn_idxs[obs_a] = get_syn_idx_of_observations(obs_a, alt_syn_obs)

                    for np_jdx_pos in range(np_idx_pos+1, len(synsets[syn_idx])):
                        np_jdx = synsets[syn_idx][np_jdx_pos]
                        for obs_b in syn_obs[(np_jdx, syn_idx)]:

                            # See whether pair satisfies constraint of appearing in different alt synsets
                            # and record pair if so.
                            if obs_b not in alt_syn_idxs:
                                alt_syn_idxs[obs_b] = get_syn_idx_of_observations(obs_b, alt_syn_obs)
                            if alt_syn_idxs[obs_a] != alt_syn_idxs[obs_b]:
                                pairs.append((obs_a, obs_b))
        if (syn_idx + 1) % 100 == 0:  # DEBUG
            print "\t" + str(syn_idx + 1) + " / " + str(len(synsets)) + " synsets processed"  # DEBUG
    return pairs


def main():

    print "reading in hypothesized and gold reconstructions..."
    f = open(FLAGS_hyp_reconstruction_infile, 'rb')
    hyp_synsets, hyp_syn_obs = pickle.load(f)
    f.close()
    f = open(FLAGS_gold_reconstruction_infile, 'rb')
    gold_synsets, gold_syn_obs = pickle.load(f)
    f.close()
    print "... done"

    # Get pairs of observations in gold reconstruction which are in the same synset but come from
    # distinct word senses, but appear in the hypothesized reconstruction in distinct synsets.
    # This explicitly tests for a polysemy + synonymy decision in gold that was not performed in hyp.
    print "getting pairs that belong together in gold but not hypothesized..."
    gold_maps = get_mismatch_maps(gold_synsets, gold_syn_obs, hyp_synsets, hyp_syn_obs)
    print "... done"

    # Get pairs of observations in hypothesized reconstruction which are in the same synset but
    # come from distinct word senses, but appear in the gold reconstruction in distinct synsets.
    # This explicitly tests for a polysemy + synonymy decision in hypothesized that was not performed in gold.
    print "getting pairs that belong together in hypothesized but not gold..."
    hyp_maps = get_mismatch_maps(hyp_synsets, hyp_syn_obs, gold_synsets, gold_syn_obs)
    print "... done"

    # Write pairs to file.
    with open(FLAGS_outfile, 'wb') as f:
        d = [gold_maps, hyp_maps]
        pickle.dump(d, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyp_reconstruction_infile', type=str, required=True,
                        help="hypothesis reconstruction structures")
    parser.add_argument('--gold_reconstruction_infile', type=str, required=True,
                        help="gold reconstruction structures")
    parser.add_argument('--outfile', type=str, required=True,
                        help="pair of gold, hypothesized pairs")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
