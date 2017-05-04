#!/usr/bin/env python
__author__ = 'jesse'
''' takes a wnid graph, a set of wnid -> text observation maps, and a w2v embedding text file and
    calculates the textual features for the wnid observations given the observed text and outputs
    a map from wnid -> text features

'''

from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import argparse
import copy
import pickle
import sys


def main():

    # constants
    kWindow = 3

    # get method
    if FLAGS_method == "k-window":  # average a window of words around wnid nps
        use_window = True
    elif FLAGS_method == "whole":  # average all wnid document words
        use_window = False
    else:
        sys.exit("invalid method")

    # read infiles
    print "reading in graph and observations..."
    f = open(FLAGS_graph_infile, 'rb')
    wnids, synsets, nps, _ = pickle.load(f)
    f.close()
    print "... read graph"
    f = open(FLAGS_text_obs_infile, 'rb')
    wnid_text = pickle.load(f)
    f.close()
    print "... read text observations"
    print "... done"

    # calculate w2v textual features from text corpus observations
    print "calculating w2v textual features from text observations..."
    try:
        model = Word2Vec.load_word2vec_format(FLAGS_w2v_vectors, binary=(".bin" == FLAGS_w2v_vectors[-4:]))
    except UnicodeDecodeError:
        model = Word2Vec.load(FLAGS_w2v_vectors)
    zeros = []
    for common_word in ['</s>', 'of', 'the', 'and', 'a']:
        if common_word in model:
            zeros = model[common_word] * 0
            break
    if len(zeros) == 0:
        sys.exit("no common word in model")
    print "... loaded w2v model"
    wnid_textf = {}
    for wnid_idx in range(0, len(wnids)):
        wnid = wnids[wnid_idx]
        if wnid in wnid_text and len(wnid_text[wnid]) > 0:
            relevant_nps = [nps[np_idx] for np_idx in synsets[wnid_idx]]
            for obs_texts in wnid_text[wnid]:
                textf = copy.copy(zeros)
                context_words = 0

                # find nps associated with wnid in texts associated with wnid and take
                # average of w2v vectors of words in window of context around those nps
                if use_window:
                    for np in relevant_nps:
                        npt = word_tokenize(np)
                        for obs_text in obs_texts:  # snippets of continuous text from corpus
                            if len(obs_text) <= len(npt):
                                continue
                            for w_idx in range(0, len(obs_text)):
                                w_jdx = w_idx + len(npt)
                                if ' '.join(obs_text[w_idx:w_jdx]) == np:
                                    b_idx = max(0, w_idx - kWindow)
                                    e_idx = min(len(npt), w_jdx + kWindow)
                                    for context_idx in range(b_idx, e_idx):
                                        if ((context_idx < w_idx or context_idx >= w_jdx) and
                                                obs_text[context_idx] in model):
                                            textf += model[obs_text[context_idx]]
                                            context_words += 1

                # take average of w2v vectors for all words in documents associated with wnid
                else:
                    for obs_text in obs_texts:
                        for w in obs_text:
                            if w in model:
                                textf += model[w]
                                context_words += 1

                # add textual feature vector (or zeros)
                if context_words > 0:
                    textf /= context_words
                if wnid not in wnid_textf:
                    wnid_textf[wnid] = []
                wnid_textf[wnid].append(textf)
    print "... done"

    # write synsets, syn_obs of induced topology
    print "writing textf to file..."
    f = open(FLAGS_outfile, 'wb')
    d = wnid_textf
    pickle.dump(d, f)
    f.close()
    print "... done"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_infile', type=str, required=True,
                        help="wnid graph used when getting observations")
    parser.add_argument('--text_obs_infile', type=str, required=True,
                        help="wnid text observations file")
    parser.add_argument('--w2v_vectors', type=str, required=True,
                        help="word2vec vectors text file")
    parser.add_argument('--method', type=str, required=True,
                        help="either 'k-window' or 'whole'")
    parser.add_argument('--outfile', type=str, required=True,
                        help="output text features from w2v")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
