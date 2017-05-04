#!/usr/bin/env python
__author__ = 'jesse'

import argparse
import pickle
from gensim import corpora, models


def main():

    # read inputs
    num_f = FLAGS_feature_space_size

    # read infiles
    print "reading in observation texts..."
    f = open(FLAGS_wnid_obs_texts_infile, 'rb')
    wnid_observations_texts = pickle.load(f)
    f.close()
    print "... done"

    # create dictionary object from texts
    print "gathering wnid_texts into flat list of texts per observation..."
    texts = []
    wnids = wnid_observations_texts.keys()
    obs_text_keys = []
    for i in range(0, len(wnids)):
        for j in range(0, len(wnid_observations_texts[wnids[i]])):
            text = [inner for outer in wnid_observations_texts[wnids[i]][j] for inner in outer]
            texts.append(text)
            obs_text_keys.append((wnids[i], j))
    print "... done"
    print "creating dictionary object from texts..."
    dictionary = corpora.Dictionary(texts)
    dictionary.save(FLAGS_support_outfile_prefix+".dict")
    print "... done"

    # create corpus object from texts and dictionary
    print "creating corpus object from dictionary and texts..."
    corpus = [dictionary.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize(FLAGS_support_outfile_prefix+".mm", corpus)
    print "... done"

    # create tfidf transformation from corpus
    print "creating tfidf transformation object..."
    tfidf = models.TfidfModel(corpus, normalize=True)
    tfidf.save(FLAGS_support_outfile_prefix+".tfidf")
    print "... done"

    # convert corpus to tfidf vectors
    print "converting corpus to tfidf vectors..."
    corpus_tfidf = tfidf[corpus]

    # perform lsi
    print "performing LSI on tfidf corpus with " + str(num_f) + " topics..."
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=num_f)
    print "... done"

    # save lsi model
    print "saving LSI model to file..."
    lsi.save(FLAGS_support_outfile_prefix+".lsi")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wnid_obs_texts_infile', type=str, required=True,
                        help="wnid observation texts file")
    parser.add_argument('--feature_space_size', type=int, required=True,
                        help="number of features to build lsi space in")
    parser.add_argument('--support_outfile_prefix', type=str, required=True,
                        help="to store dictionary, corpus, and lsi model")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
