#!/usr/bin/env python
__author__ = 'jesse'
''' takes a wnid graph, a set of wnid -> text observation maps, and a w2v embedding text file and
    calculates the textual features for the wnid observations given the observed text and outputs
    a map from wnid -> text features

'''

from gensim.models import Word2Vec
import argparse
import numpy as np
import pickle


class TextCorpusIterator:
    def __init__(self, wnid_text):
        self.wnid_text = wnid_text

    def __iter__(self):
        for wnid in self.wnid_text:
            for obs_texts in self.wnid_text[wnid]:
                for text in obs_texts:
                    print text
                    yield text

    def __len__(self):
        c = 0
        for _ in self:
            c += 1
        return c


def main():

    # read infiles
    print "reading in text observations..."
    # with open(FLAGS_text_obs_infile, 'rb') as f:
    #     wnid_text = pickle.load(f)
    wnid_text = {}
    wnid_text["wnid"] = [[["a", "sentence"], ["another", ",", "larger", "sentence"]]]
    print "... done"

    sentences = TextCorpusIterator(wnid_text)

    # build w2v model by adaptation or instantiation
    if FLAGS_w2v_vectors != "none":
        print "loading existing model..."
        model = Word2Vec()
        model.load_word2vec_format(FLAGS_w2v_vectors, binary=(".bin" == FLAGS_w2v_vectors[-4:]))
        print "... done"

        # patch old loaded model with expected data memebers
        # model.syn0_lockf = np.ones(len(model.syn0), dtype=np.float32)
        # model.corpus_count = len(sentences)
        # sentences = TextCorpusIterator(wnid_text)

        print "building vocab from additional data..."
        model.build_vocab(sentences)
        sentences = TextCorpusIterator(wnid_text)
        print "... done... training on additional data..."
        model.train(sentences)
        print "... done"
    else:
        print "instantiating new Word2Vec model from data..."
        model = Word2Vec(sentences, size=300)
        print "... done"

    # write synsets, syn_obs of induced topology
    print "writing model to file..."
    model.save(FLAGS_outfile)
    print "... done"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_obs_infile', type=str, required=True,
                        help="wnid text observations file")
    parser.add_argument('--w2v_vectors', type=str, required=True,
                        help="word2vec vectors text file")
    parser.add_argument('--outfile', type=str, required=True,
                        help="output text features from w2v")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
