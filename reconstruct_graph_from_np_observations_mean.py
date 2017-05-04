#!/usr/bin/env python
__author__ = 'jesse'

import argparse
import pickle
import numpy
from reconstruct_graph_from_np_observations import DiskDictionary


def main():

    # read pickle inputs
    with open(FLAGS_synsets, 'rb') as f:
        synsets = pickle.load(f)
    with open(FLAGS_syn_obs, 'rb') as f:
        syn_obs = pickle.load(f)
    with open(FLAGS_wnid_urls, 'rb') as f:
        wnid_urls = pickle.load(f)
        wnids = wnid_urls.keys()

    # instantiate DiskDictionaries
    max_floats = 125000000 * 1  # allow ~1GB of floats in RAM for each map
    wnid_imgfs = DiskDictionary(FLAGS_wnid_obs_infile, max_floats, wnids)
    wnid_textfs = DiskDictionary(FLAGS_wnid_textf_infile, max_floats, wnids)

    # copy param inputs
    syn_idx = FLAGS_syn_idx
    imgf_red = FLAGS_imgf_red
    textf_red = FLAGS_textf_red
    imgf_fv_size = FLAGS_imgf_fv_size
    textf_fv_size = FLAGS_textf_fv_size

    obs = []
    for np_idx in synsets[syn_idx]:
        imgf_red_init = textf_red_init = 0
        if imgf_red > 0:
            new_obs = [wnid_imgfs[entry[0]][entry[1]]
                       for entry in syn_obs[(np_idx, syn_idx)]]
            imgf_red_init = 1
        else:
            new_obs = [wnid_textfs[entry[0]][entry[1]]
                       for entry in syn_obs[(np_idx, syn_idx)]]
            textf_red_init = 1
        for _ in range(imgf_red_init, imgf_red):
            for entry_idx in range(0, len(syn_obs[(np_idx, syn_idx)])):
                entry = syn_obs[(np_idx, syn_idx)][entry_idx]
                new_obs[entry_idx].extend(wnid_imgfs[entry[0]][entry[1]])
        for _ in range(textf_red_init, textf_red):
            for entry_idx in range(0, len(syn_obs[(np_idx, syn_idx)])):
                entry = syn_obs[(np_idx, syn_idx)][entry_idx]
                new_obs[entry_idx].extend(wnid_textfs[entry[0]][entry[1]])
        obs.extend(new_obs)
    n_obs = numpy.asarray(obs)
    if len(n_obs) > 0:
        try:
            mean = numpy.mean(n_obs, axis=0)
            var = numpy.var(n_obs, axis=0)
        except TypeError as e:
            print n_obs
            print len(n_obs)
            raise e
    else:
        mean = numpy.zeros(imgf_red*imgf_fv_size + textf_red*textf_fv_size)
        var = numpy.ones(imgf_red*imgf_fv_size + textf_red*textf_fv_size)

    # write mean to pickle
    f = open(FLAGS_outfile, 'wb')
    d = [mean, var]
    pickle.dump(d, f)
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--synsets', type=str, required=True,
                        help="synsets across which to calculate means")
    parser.add_argument('--syn_obs', type=str, required=True,
                        help="indexed observations for np/syn combinations")
    parser.add_argument('--wnid_urls', type=str, required=True,
                        help="urls against which features are indexed")
    parser.add_argument('--wnid_obs_infile', type=str, required=True,
                        help="wnid observations file")
    parser.add_argument('--wnid_textf_infile', type=str, required=True,
                        help="wnid textf observations file")
    parser.add_argument('--syn_idx', type=int, required=True,
                        help="synset for which to calculate a mean")
    parser.add_argument('--imgf_red', type=int, required=True,
                        help="image feature redundancy")
    parser.add_argument('--textf_red', type=int, required=True,
                        help="text feature redundancy")
    parser.add_argument('--imgf_fv_size', type=int, required=True,
                        help="image feature vector size")
    parser.add_argument('--textf_fv_size', type=int, required=True,
                        help="text feature vector size")
    parser.add_argument('--outfile', type=str, required=True,
                        help="output pickled return values from function")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
