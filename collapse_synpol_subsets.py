#!/usr/bin/env python
__author__ = 'jesse'

import argparse
import pickle
from reconstruct_graph_from_np_observations import update_synset_structures


def main():

    print "reading in graph and reconstruction..."
    f = open(FLAGS_graph_infile, 'rb')
    _, _, nps, _ = pickle.load(f)
    f.close()
    f = open(FLAGS_reconstruction_infile, 'rb')
    synsets, syn_obs = pickle.load(f)
    f.close()
    print "... done"

    # For every synset, if its nps number more than 1 and are properly contained within another synset,
    # collapse those synsets together as a post-processing polysemy detection step.
    print "collapsing " + str(len(synsets)) + " using np subset methodology..."
    change = True
    while change:
        change = False
        synsets_to_remove = []
        synsets_to_add = []
        syn_obs_to_add = {}
        for syn_idx in range(len(synsets)):
            if len(synsets[syn_idx]) > 1 and syn_idx not in synsets_to_remove:
                for syn_jdx in range(syn_idx+1, len(synsets)):
                    if len(synsets[syn_jdx]) > 1 and syn_jdx not in synsets_to_remove:
                        if (set(synsets[syn_idx]) < set(synsets[syn_jdx]) or
                                set(synsets[syn_jdx]) < set(synsets[syn_idx])):
                            # print "\tcollapsing " + str(synsets[syn_idx]) + " with " + str(synsets[syn_jdx])  # DEBUG
                            synsets_to_remove.append(syn_idx)
                            synsets_to_remove.append(syn_jdx)
                            to_add = synsets[syn_idx][:]
                            for np_idx in synsets[syn_idx]:
                                key = (np_idx, len(synsets_to_add))
                                if key not in syn_obs_to_add:
                                    syn_obs_to_add[key] = []
                                syn_obs_to_add[key].extend(syn_obs[(np_idx, syn_idx)])
                            for np_idx in synsets[syn_jdx]:
                                if np_idx not in to_add:
                                    to_add.append(np_idx)
                                key = (np_idx, len(synsets_to_add))
                                if key not in syn_obs_to_add:
                                    syn_obs_to_add[key] = []
                                syn_obs_to_add[key].extend(syn_obs[(np_idx, syn_jdx)])
                            synsets_to_add.append(to_add)
                            break
        if len(synsets_to_remove) > 0:
            print ("\tcollapsing " + str(len(synsets_to_remove)) + " synsets and adding "
                   + str(len(synsets_to_add)) + " replacements")  # DEBUG
            change = True
            synsets, _, syn_obs = update_synset_structures(nps, synsets, synsets_to_remove,
                                                           synsets_to_add, syn_obs, syn_obs_to_add)
    print "... done; collapsed to " + str(len(synsets)) + " synsets"

    # Write pairs to file.
    print "writing out new reconstruction..."
    with open(FLAGS_outfile, 'wb') as f:
        d = [synsets, syn_obs]
        pickle.dump(d, f)
    print "... done"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_infile', type=str, required=True,
                        help="graph structures")
    parser.add_argument('--reconstruction_infile', type=str, required=True,
                        help="reconstruction structures")
    parser.add_argument('--outfile', type=str, required=True,
                        help="reconstruction outfile")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
