#!/usr/bin/env python
__author__ = 'jesse'
''' takes a wnid graph, a set of wnid -> text observation maps, a serialized lsi model, and a
    target wnid, and calculates the textual features for the wnid observations given the observed
    text and outputs a vector of text features

'''

from gensim import corpora, models
import argparse
import pickle


def main():

    # read params
    wnid = FLAGS_target_wnid
    lsi_fsize = FLAGS_lsi_fsize

    # read infiles
    print "reading in observations and lsi model..."
    f = open(FLAGS_text_obs_infile, 'rb')
    wnid_text = pickle.load(f)
    f.close()
    print "... read text observations"
    model = models.LsiModel.load(FLAGS_lsi_model)
    print "... loaded lsi model"
    dictionary = corpora.Dictionary.load(FLAGS_lsi_dictionary)
    print "... loaded lsi dictionary"
    tfidf = models.TfidfModel.load(FLAGS_tfidf_model)
    print "... loaded tfidf model"
    print "... done"

    # calculate lsi textual features from text corpus observations
    print "calculating lsi textual features from text observations..."
    textfs = []
    for obs_texts in wnid_text[wnid]:
        corpus = [dictionary.doc2bow(text) for text in obs_texts]
        corpus_tfidf = tfidf[corpus]
        textf = [0.0 for _ in range(0, lsi_fsize)]
        if len(corpus_tfidf) > 0:
            textf_m = {a: b for (a, b) in model[corpus_tfidf[0]]}
            for idx in range(1, len(corpus_tfidf)):
                for (a, b) in model[corpus_tfidf[idx]]:
                    if a not in textf_m:
                        textf_m[a] = 0
                    textf_m[a] += b
            for fidx in textf_m:
                textf[fidx] = textf_m[fidx] / float(len(corpus_tfidf))
        textfs.append(textf)
    print "... done"

    # write synsets, syn_obs of induced topology
    print "writing textfs to file..."
    with open(FLAGS_outfile, 'wb') as f:
        d = textfs
        pickle.dump(d, f)
    print "... done"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_wnid', type=str, required=True,
                        help="target wnid string")
    parser.add_argument('--text_obs_infile', type=str, required=True,
                        help="wnid text observations file")
    parser.add_argument('--lsi_dictionary', type=str, required=True,
                        help="dictionary of words used in lsi model")
    parser.add_argument('--tfidf_model', type=str, required=True,
                        help="tfidf model used by lsi")
    parser.add_argument('--lsi_model', type=str, required=True,
                        help="serialized lsi model")
    parser.add_argument('--lsi_fsize', type=int, required=True,
                        help="number of features in lsi")
    parser.add_argument('--outfile', type=str, required=True,
                        help="output text features from w2v")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
