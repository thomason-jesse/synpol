#!/usr/bin/env python
__author__ = 'jesse'

import argparse
import copy
import numpy as np
import pickle
from IPython import embed


def show_wnid(d, wnid):
    wnids, synsets, nps, polysems, wnid_urls, re_synsets, re_syn_obs, np_train_observations, distributed_suffix = d

    wnid_idx = wnids.index(wnid)
    print "synset nps: " + str([nps[np_idx] for np_idx in synsets[wnid_idx]])
    print "urls (" + str(len(wnid_urls[wnid])) + "):\n" + "\n\t".join(wnid_urls[wnid])


def get_re_synset_nps(d, re_idx):
    wnids, synsets, nps, polysems, wnid_urls, re_synsets, re_syn_obs, np_train_observations = d

    re_synset = re_synsets[re_idx]
    re_nps = []
    for np_idx in re_synset:
        re_nps.append(nps[np_idx])

    return re_nps


def calculate_overlap_matrix_column(d, re_idx):
    wnids, synsets, nps, polysems, wnid_urls, re_synsets, re_syn_obs, np_train_observations, distributed_suffix = d

    # cut synsets down to size by observing whether they actually have observations (urls)
    old_wnids = copy.deepcopy(wnids)
    old_synsets = copy.deepcopy(synsets)
    wnids = [old_wnids[wnid_idx]
             for wnid_idx in range(0, len(old_wnids))
             if old_wnids[wnid_idx] in wnid_urls]
    synsets = [old_synsets[wnid_idx]
               for wnid_idx in range(0, len(old_wnids))
               if old_wnids[wnid_idx] in wnid_urls]

    train_observations = []
    for np_idx in range(0, len(nps)):
        if np_idx in np_train_observations:
            train_observations.extend(np_train_observations[np_idx])

    col = {}
    for syn_idx in range(0, len(synsets)):
        syn_obs = [(wnids[syn_idx], obs_idx) for obs_idx in range(0, len(wnid_urls[wnids[syn_idx]]))
                   if (wnids[syn_idx], obs_idx) in train_observations]
        entry = sum([1 if obs in syn_obs else 0
                     for np_idx in re_synsets[re_idx]
                     for obs in re_syn_obs[(np_idx, re_idx)]])
        if entry > 0:
            col[syn_idx] = entry

    print col


def show_re_synset_urls(d, re_idx, np=None):
    wnids, synsets, nps, polysems, wnid_urls, re_synsets, re_syn_obs, np_train_observations, distributed_suffix = d

    re_synset = re_synsets[re_idx]
    re_nps = []
    wnid_obs_idxs = []
    for np_idx in re_synset:
        re_nps.append(nps[np_idx])
        wnid_obs_idxs.append(re_syn_obs[(np_idx, re_idx)])

    print "nps: " + str(re_nps)
    print "wnid obs urls:"
    for idx in range(0, len(re_nps)):
        if np is None or re_nps[idx] == np:
            print "\t" + re_nps[idx] + " (" + str(len(wnid_obs_idxs[idx])) + ")"
            for wnid, obs_idx in wnid_obs_idxs[idx]:
                print "\t\t" + wnid + "\t" + wnid_urls[wnid][obs_idx]


def find_all_re_synset_urls(d, re_idx):
    wnids, synsets, nps, polysems, wnid_urls, re_synsets, re_syn_obs, np_train_observations, distributed_suffix = d

    re_synset = re_synsets[re_idx]
    re_nps = []
    wnid_obs_idxs = []
    for np_idx in range(len(nps)):
        if (np_idx, re_idx) in re_syn_obs:
            if np_idx not in re_synset:
                print "WARNING: '" + nps[np_idx] + "' has observations for synset but is not in it"
            re_nps.append(nps[np_idx])
            wnid_obs_idxs.append(re_syn_obs[(np_idx, re_idx)])

    print "nps: " + str(re_nps)
    print "wnid obs urls:"
    for idx in range(0, len(re_nps)):
        print "\t" + re_nps[idx] + " (" + str(len(wnid_obs_idxs[idx])) + ")"
        for wnid, obs_idx in wnid_obs_idxs[idx]:
            print "\t\t" + wnid + "\t" + wnid_urls[wnid][obs_idx]


def find_re_synsets_with_np(d, np):
    wnids, synsets, nps, polysems, wnid_urls, re_synsets, re_syn_obs, np_train_observations, distributed_suffix = d

    np_idx = nps.index(np)
    return [re_idx for re_idx in range(0, len(re_synsets)) if np_idx in re_synsets[re_idx]]


def get_re_synset_size_stats(d, obs_level=True):
    wnids, synsets, nps, polysems, wnid_urls, re_synsets, re_syn_obs, np_train_observations, distributed_suffix = d

    sizes = []
    for re_idx in range(0, len(re_synsets)):
        size = 0
        if obs_level:
            for np_idx in re_synsets[re_idx]:
                size += len(re_syn_obs[(np_idx, re_idx)])
        else:
            size += len(re_synsets[re_idx])
        sizes.append(size)

    print "count: " + str(len(sizes))
    print "min: " + str(min(sizes))
    print "max: " + str(max(sizes)) + "; idx=" + str(np.where(np.asarray(sizes) == max(sizes)))
    print "avg: " + str(sum(sizes) / float(len(sizes)))
    ss = set(sorted(sizes))
    size_pairs = []
    for ssize in ss:
        size_pairs.append((ssize, sum([1 for size in sizes if size == ssize])))
    return size_pairs


def get_synpol_nps(d, minpolycount=2, minsyncount=2):
    wnids, synsets, nps, polysems, wnid_urls, re_synsets, re_syn_obs, np_train_observations, distributed_suffix = d

    synpol_nps = []
    for np_idx in polysems:
        polycount = 0
        syncount = 0
        for wnid in polysems[np_idx]:
            if wnid in wnid_urls:
                polycount += 1
                if len(synsets[wnids.index(wnid)]) > 1:
                    syncount += 1
            if polycount >= minpolycount and syncount >= minsyncount:
                synpol_nps.append(nps[np_idx])
                break
    return synpol_nps


def get_train_obs_for_nps(d, target_nps):
    wnids, synsets, nps, polysems, wnid_urls, re_synsets, re_syn_obs, np_train_observations, distributed_suffix = d

    target_np_idxs = [nps.index(np) for np in target_nps]
    new_obs = []
    for idx in target_np_idxs:
        new_obs.extend(np_train_observations[idx])
    return new_obs


def analyze_text_corpus(d):
    wnids, synsets, nps, polysems, wnid_urls, re_synsets, re_syn_obs, np_train_observations, distributed_suffix = d

    wnids = wnid_urls.keys()

    # Count texts and lengths
    wnids_missing = 0
    num_wnids = 0
    docs_per_wnid = 0
    avg_doc_length = 0
    for idx in range(len(wnids)):

        fn = str(idx) + "_" + distributed_suffix + "_texts.pickle"
        try:
            with open(fn, 'rb') as f:
                t = pickle.load(f)
                num_wnids += 1
                docs_per_wnid += len(t[wnids[idx]])
                avg_doc_length += sum([len(doc) for doc in t[wnids[idx]]])
        except IOError:
            wnids_missing += 1

        if idx % 100 == 0:
            print "... " + str(idx) + " / " + str(len(wnids))

    avg_doc_length /= float(docs_per_wnid)
    docs_per_wnid /= float(num_wnids)

    print "wnids: " + str(num_wnids)
    print "wnids missing: " + str(wnids_missing)
    print "avg docs per wnid: " + str(docs_per_wnid)
    print "avg doc length: " + str(avg_doc_length)


def main():

    # read infiles
    print "reading in graph, observations, and reconstruction..."
    f = open(FLAGS_graph_infile, 'rb')
    wnids, synsets, nps, polysems = pickle.load(f)
    f.close()
    f = open(FLAGS_wnid_obs_url_infile, 'rb')
    wnid_urls = pickle.load(f)
    f.close()
    f = open(FLAGS_reconstruction_infile, 'rb')
    re_synsets, re_syn_obs = pickle.load(f)
    f.close()
    f = open(FLAGS_np_train_obs, 'rb')
    np_train_observations = pickle.load(f)
    f.close()
    distributed_suffix = FLAGS_distributed_suffix
    print "... done"

    d = [wnids, synsets, nps, polysems, wnid_urls,
         re_synsets, re_syn_obs, np_train_observations, distributed_suffix]

    embed()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_infile', type=str, required=True,
                        help="wnid graph used to construct observations")
    parser.add_argument('--wnid_obs_url_infile', type=str, required=True,
                        help="wnid observations url file (faster to load; don't actually need numbers)")
    parser.add_argument('--np_train_obs', type=str, required=False,
                        help="observations to consider")
    parser.add_argument('--reconstruction_infile', type=str, required=True,
                        help="reconstruction structures to be tested")
    parser.add_argument('--distributed_suffix', type=str, required=True,
                        help="suffix for distributed image features and text data")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
