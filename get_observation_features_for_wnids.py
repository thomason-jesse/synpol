#!/usr/bin/env python
__author__ = 'jesse'
''' pass this a synpol data graph pickle

    outputs map from wnids to observation feature vectors
    should be compatible across data graphs due to
    blank observation set means no image data available for synset
'''

import argparse
import pickle
import os
import matplotlib.pyplot as plt
import subprocess
import sys

sys.path.append('/scratch/cluster/jesse/caffe/examples/feature_extraction/')
sys.path.append('/scratch/cluster/jesse/caffe/python/')
import FeatureExtractor

devnull = open(os.devnull, 'w')


def process_observations(img_paths_fn, img_urls_fn):
    observations = []
    print "processing observations ..."
    cmd = "(cd /scratch/cluster/jesse/caffe/examples/feature_extraction/ ; " + \
        "python extract_vgg_features.py /scratch/cluster/jesse/synpol/"+img_paths_fn + \
          " /scratch/cluster/jesse/synpol/"+img_paths_fn+".csv)"
    # print "... cmd: "+str(cmd)  # DEBUG
    os.system(cmd)
    # print "... executed"
    f = open(img_paths_fn+'.csv', 'r')
    lines = f.readlines()
    f.close()
    f = open(img_urls_fn, 'r')
    url_lines = f.readlines()
    f.close()
    print "..... got back "+str(len(lines))+" image feature vectors"
    for line_idx in range(0, len(lines)):
        line = lines[line_idx]
        url_line = url_lines[line_idx]
        im_fn = line.split(',')[0]
        wnid = im_fn.split('.')[1]
        url = url_line.strip()
        observations.append([wnid, url, [float(d) for d in line.split(',')[1:]]])
        os.system("rm "+str(im_fn)+" > /dev/null 2> /dev/null")
    os.system("rm "+img_paths_fn+".csv")
    print "... done"
    return observations


def main():

    observation_batch_size = 100  # how many images to keep on disk before running NN
    thread_batch_size = 10  # how many images to try to curl at the same time
    max_observations_per_noun_phrase = FLAGS_observations_per_np

    fe = FeatureExtractor.FeatureExtractor("/scratch/cluster/jesse/caffe/models/vgg/VGG_ILSVRC_16_layers.caffemodel",
                                           "/scratch/cluster/jesse/caffe/models/vgg/vgg_orig_16layer.deploy.prototxt",
                                           -1)
    wnid_observations = {}
    wnid_urls = {}

    # read in synpol data graph structures
    print "reading synpol data graph pickle"
    f = open(FLAGS_data_infile, 'rb')
    wnids, synsets, _, _ = pickle.load(f)
    f.close()
    print "... done"

    # for each wnid, download all images and extract features
    print "downloading images and extracting features..."
    wnid_idx = 0
    observations_to_process = 0
    img_paths_fn = FLAGS_obs_outfile+".imgs"
    img_urls_fn = FLAGS_obs_outfile+".imgs.urls"
    f_out = open(img_paths_fn, 'w')
    f_url_out = open(img_urls_fn, 'w')
    while wnid_idx < len(wnids):
        target_num_observations = max_observations_per_noun_phrase*len(synsets[wnid_idx])
        wnid = wnids[wnid_idx]
        print "... gathering for wnid '"+str(wnid)+"'... "
        cmd = "wget http://www.image-net.org/api/text/imagenet.synset.geturls?wnid="+str(wnid) + \
              " 2> /dev/null"
        # print "... cmd: "+str(cmd)
        os.system(cmd)
        # print "...... executed"
        f = open("imagenet.synset.geturls?wnid="+str(wnid), 'r')
        lines = f.readlines()
        if len(lines) <= 1 or lines[0] == "The synset is not ready yet. Please stay tuned!":
            wnid_idx += 1
            f.close()
            os.system("rm imagenet.synset.geturls?wnid="+str(wnid)+" > /dev/null 2> /dev/null")
            continue
        observations_for_wnid = 0
        line_idx = 0
        curl_threads = []
        curl_fns = []
        curl_urls = []
        # print "...... launching initial batch of curl threads"
        for pdx in range(0, thread_batch_size):  # open initial curl threads
            url = lines[line_idx].strip().lower()
            im_fn = os.path.join(os.getcwd(), "img."+str(wnid)+"."+str(line_idx))
            cmd = ["wget", url, "-O", im_fn]
            # print "...... cmd="+str(cmd)
            p = subprocess.Popen(cmd, stdout=devnull, stderr=subprocess.STDOUT)
            curl_threads.append(p)
            curl_fns.append(im_fn)
            curl_urls.append(url)
            line_idx += 1
            if line_idx == len(lines):
                break
        # print "...... polling curl threads"
        while 1 in [0 if t is None else 1 for t in curl_threads] and \
                observations_for_wnid < target_num_observations:
            # continue polling threads and add successfully curled images to those to be processed
            # when a thread finishes, start a new download
            for pdx in range(0, len(curl_threads)):
                if curl_threads[pdx] is not None and curl_threads[pdx].poll() is not None:
                    im_fn = curl_fns[pdx]
                    try:
                        _ = plt.imread(im_fn)  # file exists and can later be read by matplotlib in NN
                        _ = fe.preprocess_image(im_fn)  # file can be preprocessed by caffe
                        # print "...... adding "+im_fn+" to list of those to be processed"
                        f_out.write(im_fn+'\n')
                        f_url_out.write(curl_urls[pdx]+'\n')
                        observations_to_process += 1
                        observations_for_wnid += 1
                        if observations_to_process >= observation_batch_size:  # process batch in NN
                            observations_to_process = 0
                            f_out.close()
                            f_url_out.close()
                            observations = process_observations(img_paths_fn, img_urls_fn)
                            if len(observations) < observations_to_process:
                                print "WARNING: got "+str(len(observations))+" observations of"+\
                                    str(observations_to_process)+" queued for processing"
                            observations_to_process = 0
                            for obs_wnid, obs_url, obs in observations:
                                if obs_wnid not in wnid_observations:
                                    wnid_observations[obs_wnid] = []
                                    wnid_urls[obs_wnid] = []
                                wnid_observations[obs_wnid].append(obs)
                                wnid_urls[obs_wnid].append(obs_url)
                            # pickle output files
                            # print "writing output pickle"
                            # pickle_f = open(FLAGS_obs_outfile, 'wb')
                            # d = wnid_observations
                            # pickle.dump(d, pickle_f)
                            # pickle_f.close()
                            # pickle_f = open(FLAGS_url_outfile, 'wb')
                            # d = wnid_urls
                            # pickle.dump(d, pickle_f)
                            # pickle_f.close()
                            # print "...... done"
                            f_out = open(img_paths_fn, 'w')
                            f_url_out = open(img_urls_fn, 'w')
                    except (IOError, ValueError, IndexError):
                        os.system("rm "+str(im_fn)+" > /dev/null 2> /dev/null")
                    curl_threads[pdx] = None
                    curl_fns[pdx] = None
                    curl_urls[pdx] = None
                    if observations_for_wnid == target_num_observations:
                        break  # leave loop if we've got enough images now
                    # create new thread to take pdx place
                    if line_idx < len(lines):
                        url = lines[line_idx].strip().lower()
                        im_fn = os.path.join(os.getcwd(), "img."+str(wnid)+"."+str(line_idx))
                        cmd = ["wget", url, "-O", im_fn]
                        p = subprocess.Popen(cmd, stdout=devnull, stderr=subprocess.STDOUT)
                        curl_threads[pdx] = p
                        curl_fns[pdx] = im_fn
                        curl_urls[pdx] = url
                        line_idx += 1

        os.system("rm imagenet.synset.geturls?wnid="+str(wnid)+" > /dev/null 2> /dev/null")

        # clean up threads images, if any; let threads that haven't terminated yet die in their own time
        for pdx in range(0, len(curl_threads)):
            if curl_fns[pdx] is not None:
                os.system("rm "+str(curl_fns[pdx])+" > /dev/null 2> /dev/null")

        wnid_idx += 1

    f_out.close()
    f_url_out.close()
    observations = process_observations(img_paths_fn, img_urls_fn)
    for obs_wnid, obs_url, obs in observations:
        if obs_wnid not in wnid_observations:
            wnid_observations[obs_wnid] = []
            wnid_urls[obs_wnid] = []
        wnid_observations[obs_wnid].append(obs)
        wnid_urls[obs_wnid].append(obs_url)

    # pickle output files
    print "writing final output pickles"
    f = open(FLAGS_obs_outfile, 'wb')
    d = wnid_observations
    pickle.dump(d, f)
    f.close()
    f = open(FLAGS_url_outfile, 'wb')
    d = wnid_urls
    pickle.dump(d, f)
    f.close()
    print "... done"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_infile', type=str, required=True,
                        help="input synpol data graph pickle")
    parser.add_argument('--observations_per_np', type=int, required=True,
                        help="max number of observations to gather per noun phrase in each wnid")
    parser.add_argument('--obs_outfile', type=str, required=True,
                        help="output pickle of wnid->observations features map")
    parser.add_argument('--url_outfile', type=str, required=True,
                        help="output pickle of wnid->observation_urls map")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
