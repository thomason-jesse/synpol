#!/usr/bin/env python
__author__ = 'jesse'
''' pass this a text file of wnid, synset pairs

    outputs synonymy/polysemy extracted from synsets
'''

import argparse
import pickle
import os


def main():

    # read in wnid for which features are available
    try:
        feat_av = []
        f = open(FLAGS_featav_infile, 'r')
        for line in f.readlines():
            feat_av.append(line.strip().lower())
        f.close()
    except TypeError:
        feat_av = None

    # read in blacklisted wnids
    blacklist = []
    try:
        f = open(FLAGS_blacklist, 'r')
        for line in f.readlines():
            blacklist.append(line.strip().lower())
        f.close()
    except TypeError:
        pass

    # read in pairs
    wnids = []  # list of wnids of synsets
    synsets = []  # list of synsets; synsets are lists of noun phrase ids and serve as synonymy relationships
    noun_phrases = []  # list of noun phrases used to compose synsets as lists of words

    print "reading in wnid, synset pairs"
    f = open(FLAGS_words_infile, 'r')
    lines = f.readlines()
    p = 0.1
    for line_idx in range(0, len(lines)):
        line = lines[line_idx]
        wnid, synset_str = line.strip().lower().split('\t')
        if wnid in blacklist:
            print "skipping blacklisted "+wnid  # DEBUG
            continue

        include = True
        if FLAGS_leaves_only:
            os.system("wget http://www.image-net.org/api/text/wordnet.structure.hyponym?wnid="+str(wnid) +
                      " 2> /dev/null")
            child_f = open("wordnet.structure.hyponym?wnid="+str(wnid), 'r')
            if len(child_f.readlines()) > 1:
                include = False
            child_f.close()
            os.system("rm wordnet.structure.hyponym?wnid="+str(wnid))
        if FLAGS_images_only:
            os.system("wget http://www.image-net.org/api/text/imagenet.synset.geturls?wnid="+str(wnid) +
                      " 2> /dev/null")
            im_f = open("imagenet.synset.geturls?wnid="+str(wnid), 'r')
            im_lines = im_f.readlines()
            if len(im_lines) > 0:
                if im_lines[0] == "The synset is not ready yet. Please stay tuned!":
                    include = False
            else:
                include = False
            im_f.close()
            os.system("rm imagenet.synset.geturls?wnid="+str(wnid))

        if include:
            if feat_av is None or wnid in feat_av:
                wnids.append(wnid)
                noun_phrase_strs = synset_str.split(', ')
                synset = []
                for nps in noun_phrase_strs:
                    if nps not in noun_phrases:
                        noun_phrases.append(nps)
                    synset.append(noun_phrases.index(nps))
                synsets.append(synset)

        if line_idx / float(len(lines)) > p:
            print "... processed "+str(p*len(lines))+" / "+str(len(lines))+" lines"
            p += 0.1
    f.close()
    print "... done; got "+str(len(wnids))+" synsets composed of "+str(len(noun_phrases))+" noun phrases"

    # calculate polysemy relationships between noun phrases
    print "calculating polysemy relationships between noun phrases"
    polysems = {}  # noun phrase key indexes to wnid list
    for idx in range(0, len(synsets)-1):
        for np_idx_a in range(0, len(synsets[idx])):
            if synsets[idx][np_idx_a] not in polysems:
                polysems[synsets[idx][np_idx_a]] = []
            polysems[synsets[idx][np_idx_a]].append(wnids[idx])
    singletons = []
    for np_idx in polysems:
        if len(polysems[np_idx]) <= 1:
            singletons.append(np_idx)
    for np_idx in singletons:
        del polysems[np_idx]
    print "... done; got "+str(len(polysems))+" polysemous sets of words"

    # pickle output files
    print "writing output pickle"
    f = open(FLAGS_outfile, 'w')
    d = [wnids, synsets, noun_phrases, polysems]
    pickle.dump(d, f)
    f.close()
    print "... done"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--words_infile', type=str, required=True,
                        help="input textfile wnid, synset pairs")
    parser.add_argument('--featav_infile', type=str, required=False,
                        help="input textfile wnid for which external features are available")
    parser.add_argument('--leaves_only', type=bool, default=False, required=False,
                        help="only consider synsets which have no children")
    parser.add_argument('--images_only', type=bool, default=False, required=False,
                        help="only consider synsets which have images")
    parser.add_argument('--blacklist', type=str, required=False,
                        help="list of wnids not to be added")
    parser.add_argument('--outfile', type=str, required=True,
                        help="output pickle of synonymy/polysemy graphs extracted")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
