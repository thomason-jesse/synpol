#!/usr/bin/env python
__author__ = 'jesse'
''' pass this a synpol data graph pickle

    outputs a synpol data graph pickle that has been
    trimmed to maximize and balance synonymy and
    polysemy among remaining noun phrases
'''

import argparse
import pickle
import random


def main():

    # read in optional flags
    try:
        size = FLAGS_size
    except NameError:
        size = None
    try:
        blacklist_fn = FLAGS_blacklist_graph
        print "reading blacklist graph data pickle..."
        f = open(blacklist_fn, 'rb')
        blacklist_wnids, _, _, _ = pickle.load(f)
        f.close()
        print "... done"
    except NameError:
        blacklist_wnids = []

    # read in synpol data graph structures
    print "reading synpol data graph pickle..."
    f = open(FLAGS_data_infile, 'rb')
    orig_wnids, orig_synsets, orig_noun_phrases, orig_polysems = pickle.load(f)
    f.close()
    print "... done"

    # initial stats
    print "synsets max: "+str(max([len(nps) for nps in orig_synsets]))
    print "synsets avg: "+str(sum([len(nps) for nps in orig_synsets]) / float(len(orig_synsets)))
    print "synsets min: "+str(min([len(nps) for nps in orig_synsets]))
    print "polysemy max: "+str(max([len(orig_polysems[np]) for np in orig_polysems]))
    print "polysemy avg: "+str(sum([len(orig_polysems[np]) for np in orig_polysems]) / float(len(orig_polysems.keys())))
    print "polysemy min: "+str(min([len(orig_polysems[np]) for np in orig_polysems]))

    # count
    print "calculating new stats of interest..."
    num_syn = 0
    num_pol = 0
    num_both = 0
    num_neither = 0
    for np_idx in range(0, len(orig_noun_phrases)):
        syn = False
        pol = np_idx in orig_polysems
        for syn_idx in range(0, len(orig_synsets)):
            if np_idx in orig_synsets[syn_idx] and len(orig_synsets[syn_idx]) > 1:
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

    # convert to synonymy and polysemy graphs
    print "constructing graph representation..."
    nodes = []
    syn_edges = []
    pol_edges = []
    degree_zero_nodes = []
    connected_nodes = []
    for idx in range(0, len(orig_wnids)):
        if orig_wnids[idx] in blacklist_wnids:
            continue
        for np_idx_pos in range(0, len(orig_synsets[idx])):
            np_idx = orig_synsets[idx][np_idx_pos]
            n = (idx, np_idx)
            nodes.append(n)
            dz = True if n not in connected_nodes else False
            for np_jdx_pos in range(np_idx_pos+1, len(orig_synsets[idx])):
                np_jdx = orig_synsets[idx][np_jdx_pos]
                if np_idx != np_jdx:
                    m = (idx, np_jdx)
                    e = (n, m)
                    syn_edges.append(e)
                    connected_nodes.extend([n, m])
                    dz = False
            for jdx in range(idx+1, len(orig_wnids)):
                if orig_wnids[jdx] in blacklist_wnids:
                    continue
                if np_idx in orig_synsets[jdx]:
                    m = (jdx, np_idx)
                    e = (n, m)
                    pol_edges.append(e)
                    connected_nodes.extend([n, m])
                    dz = False
            if dz:
                degree_zero_nodes.append(n)
    print "... done; "+str(len(nodes))+" nodes with "+str(len(syn_edges)) + \
          " syn edges and "+str(len(pol_edges))+" pol edges and " + \
          str(len(degree_zero_nodes))+" isolated nodes"

    # naive heuristic, take one syn edge, one pol edge, and one isolated node at a time
    print "running naive sub-graph induction procedure..."
    pol_edges_to_get = pol_edges[:]
    random.shuffle(pol_edges_to_get)
    syn_edges_to_get = syn_edges[:]
    random.shuffle(syn_edges_to_get)
    random.shuffle(degree_zero_nodes)
    subgraph_nodes = []
    subgraph_syn_edges = []
    subgraph_pol_edges = []
    subgraph_syn_nodes = []
    subgraph_pol_nodes = []
    subgraph_zer_nodes = []
    added_anything = True
    while added_anything:
        added_anything = False
        max_nodes = max([len(subgraph_pol_nodes), len(subgraph_syn_nodes), len(subgraph_zer_nodes)]) \
            if size is None else size
        syn_edge = syn_edges_to_get.pop() if len(subgraph_syn_nodes) <= max_nodes else None
        pol_edge = pol_edges_to_get.pop() if len(subgraph_pol_nodes) <= max_nodes else None
        degree_zero_node = degree_zero_nodes.pop() if len(subgraph_zer_nodes) <= max_nodes else None
        to_add = []
        if syn_edge is not None:
            to_add.extend([syn_edge[0], syn_edge[1]])
            subgraph_syn_edges.append(syn_edge)
            for n in [syn_edge[0], syn_edge[1]]:
                if n not in subgraph_syn_nodes:
                    subgraph_syn_nodes.append(n)
        if pol_edge is not None:
            to_add.extend([pol_edge[0], pol_edge[1]])
            subgraph_pol_edges.append(pol_edge)
            for n in [pol_edge[0], pol_edge[1]]:
                if n not in subgraph_pol_nodes:
                    subgraph_pol_nodes.append(n)
        if degree_zero_node is not None:
            to_add.append(degree_zero_node)
            subgraph_zer_nodes.append(degree_zero_node)
        for n in to_add:
            if n not in subgraph_nodes:
                subgraph_nodes.append(n)
                added_anything = True
    print "... done; "+str(len(subgraph_nodes))+" nodes with "+str(len(subgraph_syn_edges)) + \
          " syn edges and "+str(len(subgraph_pol_edges))+" pol edges and " + \
          str(len(subgraph_zer_nodes))+" isolated nodes"

    # reconstruct the wnids, synsets, noun phrases, and polysems lists with new graph
    print "reconstructing wnid_graph structures given new subgraph"
    wnids = []
    synsets = []
    noun_phrases = []
    polysems = {}
    for n in subgraph_nodes:
        orig_wnid_idx, orig_np_idx = n
        if orig_noun_phrases[orig_np_idx] not in noun_phrases:
            noun_phrases.append(orig_noun_phrases[orig_np_idx])
        if orig_wnids[orig_wnid_idx] not in wnids:
            wnids.append(orig_wnids[orig_wnid_idx])
            synsets.append([noun_phrases.index(orig_noun_phrases[orig_np_idx])])
    for e in subgraph_syn_edges:
        idx = wnids.index(orig_wnids[e[0][0]])
        for orig_np_idx in [e[0][1], e[1][1]]:
            np_idx = noun_phrases.index(orig_noun_phrases[orig_np_idx])
            if np_idx not in synsets[idx]:
                synsets[idx].append(np_idx)
    for e in subgraph_pol_edges:
        np_idx = noun_phrases.index(orig_noun_phrases[e[0][1]])
        if np_idx not in polysems:
            polysems[np_idx] = []
        for orig_wnid_idx in [e[0][0], e[1][0]]:
            if orig_wnids[orig_wnid_idx] not in polysems[np_idx]:
                polysems[np_idx].append(orig_wnids[orig_wnid_idx])
            idx = wnids.index(orig_wnids[orig_wnid_idx])
            if np_idx not in synsets[idx]:
                synsets[idx].append(np_idx)
    print "... done"

    # count
    print "calculating new stats of interest..."
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

    # show results and stats
    print "wnids reduced from "+str(len(orig_wnids))+" to "+str(len(wnids))
    print "synsets reduced from "+str(len(orig_synsets))+" to "+str(len(synsets))
    print "noun phrases reduced from "+str(len(orig_noun_phrases))+" to "+str(len(noun_phrases))
    print "polysems reduced from "+str(len(orig_polysems.keys()))+" to "+str(len(polysems.keys()))
    print "synsets max: "+str(max([len(nps) for nps in synsets]))
    print "synsets avg: "+str(sum([len(nps) for nps in synsets]) / float(len(synsets)))
    print "synsets min: "+str(min([len(nps) for nps in synsets]))
    print "polysemy max: "+str(max([len(polysems[np]) for np in polysems]))
    print "polysemy avg: "+str(sum([len(polysems[np]) for np in polysems]) / float(len(polysems.keys())))
    print "polysemy min: "+str(min([len(polysems[np]) for np in polysems]))

    # pickle output files
    print "writing output pickle"
    f = open(FLAGS_outfile, 'w')
    d = [wnids, synsets, noun_phrases, polysems]
    pickle.dump(d, f)
    f.close()
    print "... done"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_infile', type=str, required=True,
                        help="input synpol data graph pickle")
    parser.add_argument('--blacklist_graph', type=str, required=False,
                        help="data graph pickle with nodes not to be included here")
    parser.add_argument('--size', type=int, required=False,
                        help="number of nodes of each class to cap at when making subset")
    parser.add_argument('--outfile', type=str, required=True,
                        help="output synpol data graph pickle")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
