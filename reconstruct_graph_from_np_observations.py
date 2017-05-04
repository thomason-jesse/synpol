#!/usr/bin/env python
__author__ = 'jesse'
''' give this a set of np_observations

    outputs synsets and observation pairing attempting to reconstruct original wnid_graph based on observations alone

'''

from gap_statistic_functions import *
from scipy.spatial.distance import cosine
import argparse
import copy
import numpy as np
import math
import pickle
import operator
import os
import sys
import time


class DiskDictionary:
    def __init__(self, source_suffix, max_floats, wnids):
        self.source_suffix = source_suffix
        self.max_floats = max_floats
        self.wnids = wnids
        self.d = {}
        self.accessed = {}
        self.curr_floats = 0

    def __getitem__(self, key):

        # need to load key from disk
        if key not in self.d:

            # load from disk into map
            key_idx = self.wnids.index(key)
            try:
                with open(str(key_idx) + "_" + self.source_suffix) as f:
                    new_data = pickle.load(f)
            except (IOError, EOFError):
                raise KeyError(key_idx)
            if type(new_data) is dict:
                for new_key in new_data:
                    self.d[new_key] = new_data[new_key]
                    self.curr_floats += len(new_data[new_key])*len(new_data[new_key][0])
                    if new_key not in self.accessed:
                        self.accessed[new_key] = 0
            else:
                self.d[key] = new_data
                self.curr_floats += len(new_data)
                if key not in self.accessed:
                    self.accessed[key] = 0

            # delete least recently accessed if over memory budget
            while self.curr_floats > self.max_floats:
                # print ("WARNING: DiskDictionary with " + self.source_suffix + " is over-budget with " +
                #        str(len(self.d.keys())) + " wnids loaded")  # DEBUG
                (del_key, _) = sorted(self.accessed.items(), key=operator.itemgetter(1), reverse=True)[0]
                if del_key == key:
                    print "... WARNING: DiskDictionary tried to remove current key; ignoring"  # DEBUG
                    break
                self.curr_floats -= len(self.d[del_key])*len(self.d[del_key][0])
                del self.d[del_key]
                del self.accessed[del_key]

        # update access times and return data
        for loaded_key in self.d:
            self.accessed[loaded_key] += 1
            if loaded_key == key:
                self.accessed[loaded_key] = 0
        return self.d[key]

# estimated from blacklist
word_senses_per_cluster_dev = 1.78126117241
alpha_pol_density_dev = 0.357746666085
alpha_syn_density_dev = 6827.31897579
max_polysemy_dev = 32

# globals
_CONDOR_MAX_JOBS = 1000


def update_synset_structures(nps, synsets, synsets_to_remove, synsets_to_add, syn_obs, syn_obs_to_add):
    synsets = [synsets[sdx] for sdx in range(0, len(synsets)) if sdx not in synsets_to_remove]
    for syn_idx in synsets_to_remove:
        for np_idx in range(0, len(nps)):
            if (np_idx, syn_idx) in syn_obs:
                del syn_obs[(np_idx, syn_idx)]
    syn_obs_old = copy.deepcopy(syn_obs)
    syn_obs = {}
    for key in syn_obs_old:
        syn_idx = key[1]
        for syn_jdx in synsets_to_remove:
            if syn_jdx < key[1]:
                syn_idx -= 1
        syn_obs[(key[0], syn_idx)] = syn_obs_old[key]
    last_syn_idx = len(synsets)
    synsets.extend(synsets_to_add)
    for key in syn_obs_to_add:
        # update to new indexing of synsets after deletions and additions
        key_updated = (key[0], key[1]+last_syn_idx)
        if key_updated not in syn_obs:
            syn_obs[key_updated] = []
        syn_obs[key_updated].extend(syn_obs_to_add[key])
    return synsets, last_syn_idx, syn_obs


def get_synset_means(outf, urls_fn, synsets, syn_obs, wnid_imgfs_fn, wnid_textfs_fn,
                     imgf_red, textf_red, imgf_fv_size, textf_fv_size):
    # write input files needed
    synsets_fn = outf + ".synsets"
    syn_obs_fn = outf + ".syn_obs"
    with open(synsets_fn, 'wb') as f:
        pickle.dump(synsets, f)
    with open(syn_obs_fn, 'wb') as f:
        pickle.dump(syn_obs, f)

    unlaunched_jobs = range(0, len(synsets))
    remaining_jobs = []
    mean_dict = {}
    var_dict = {}
    while len(unlaunched_jobs) > 0 or len(remaining_jobs) > 0:

        # launch jobs for each synset
        newly_launched = []
        for syn_idx in unlaunched_jobs:
            if len(remaining_jobs) >= _CONDOR_MAX_JOBS:
                break
            cmd = ("condorify_gpu_email python reconstruct_graph_from_np_observations_mean.py " +
                   "--synsets " + synsets_fn + " " +
                   "--syn_obs " + syn_obs_fn + " " +
                   "--wnid_urls " + urls_fn + " " +
                   "--wnid_obs_infile " + wnid_imgfs_fn + " " +
                   "--wnid_textf_infile " + wnid_textfs_fn + " " +
                   "--syn_idx " + str(syn_idx) + " " +
                   "--imgf_red " + str(imgf_red) + " " +
                   "--textf_red " + str(textf_red) + " " +
                   "--imgf_fv_size " + str(imgf_fv_size) + " " +
                   "--textf_fv_size " + str(textf_fv_size) + " " +
                   "--outfile " + outf + "." + str(syn_idx) + ".mean " +
                   outf + "." + str(syn_idx) + ".mean.log")
            os.system(cmd)
            remaining_jobs.append(syn_idx)
            newly_launched.append(syn_idx)
        unlaunched_jobs = [job for job in unlaunched_jobs if job not in newly_launched]
        if len(newly_launched) > 0:
            print ("......... " + str(len(unlaunched_jobs)) + " jobs remain after launching "
                   + str(len(newly_launched)))

        # poll for completed jobs and store means
        newly_finished = []
        for syn_idx in remaining_jobs:
            try:
                pf = outf + "." + str(syn_idx) + ".mean"
                with open(pf, 'rb') as f:
                    mean, var = pickle.load(f)
                newly_finished.append(syn_idx)
                mean_dict[syn_idx] = mean
                var_dict[syn_idx] = var
                os.system("rm " + pf)
                os.system("rm " + pf + ".log")
                os.system("rm err."+pf.replace("/", "-")+".log")
            except:
                pass
        remaining_jobs = [syn_idx for syn_idx in remaining_jobs if syn_idx not in newly_finished]
        if len(newly_finished) > 0:
            print ("......... " + str(len(remaining_jobs)) + " jobs remain after gathering " +
                   str(len(newly_finished)))

        if len(remaining_jobs) >= _CONDOR_MAX_JOBS or len(newly_launched) == 0:
            time.sleep(60)

    # convert dict to vector
    means = [mean_dict[syn_idx] for syn_idx in range(0, len(synsets))]
    vars = [var_dict[syn_idx] for syn_idx in range(0, len(synsets))]

    # clean up input files
    os.system("rm " + synsets_fn)
    os.system("rm " + syn_obs_fn)

    return means, vars


def complete_synset_vars(synsets, syn_obs, vars):
    print "complete_synset_vars: getting num observations list..."
    num_obs = [len(syn_obs[(np_idx, syn_idx)])
               for syn_idx in range(len(synsets))
               for np_idx in synsets[syn_idx]]
    print "... done"

    print "complete_synset_vars: calculating averages and completing variance vectors..."
    uniform = 0
    for f_idx in range(0, len(vars[0])):
        nonzero_syn_idxs = np.nonzero([vars[syn_idx][f_idx] for syn_idx in range(len(synsets))])[0]
        num_obs_for_f = sum([num_obs[syn_idx] for syn_idx in nonzero_syn_idxs])
        if num_obs_for_f > 0:
            avg = (sum([vars[syn_idx][f_idx]*num_obs[syn_idx] for syn_idx in nonzero_syn_idxs]) /
                   float(num_obs_for_f))
        else:
            avg = 1.0
            uniform += 1

        for syn_idx in range(len(synsets)):
            if syn_idx in nonzero_syn_idxs:
                continue
            else:
                vars[syn_idx][f_idx] = avg
    print "... done; " + str(uniform) + " features had no nonzero vars of " + str(len(vars[0])) + " total features"

    return vars


def get_mean_connectivity_matrix(outf, means_fn, synsets, ha_conn_penalty=None, vars_fn=None):
    # write input files needed
    synsets_fn = outf + ".synsets"
    with open(synsets_fn, 'wb') as f:
        pickle.dump(synsets, f)

    unlaunched_jobs = range(0, len(synsets))
    remaining_jobs = []
    conn = numpy.zeros([len(synsets), len(synsets)])
    while len(unlaunched_jobs) > 0 or len(remaining_jobs) > 0:

        # launch jobs
        newly_launched = []
        for syn_idx in unlaunched_jobs:
            if len(remaining_jobs) >= _CONDOR_MAX_JOBS:
                break
            cmd = ("condorify_gpu_email python reconstruct_graph_from_np_observations_conn.py " +
                   "--synsets " + synsets_fn + " " +
                   "--means " + means_fn + " ")
            if vars_fn is not None:
                   cmd += "--vars " + vars_fn + " "
            cmd += "--syn_idx " + str(syn_idx) + " "
            if ha_conn_penalty is not None:
                cmd += ("--penalty " + str(ha_conn_penalty) + " " +
                        "--outfile " + outf + "." + str(syn_idx) + ".conn" + str(ha_conn_penalty) + " " +
                        outf + "." + str(syn_idx) + ".conn" + str(ha_conn_penalty) + ".log")
            else:
                cmd += ("--outfile " + outf + "." + str(syn_idx) + ".conn " +
                        outf + "." + str(syn_idx) + ".conn.log")
            os.system(cmd)
            remaining_jobs.append(syn_idx)
            newly_launched.append(syn_idx)
        unlaunched_jobs = [job for job in unlaunched_jobs if job not in newly_launched]
        if len(newly_launched) > 0:
            print ("......... " + str(len(unlaunched_jobs)) + " jobs remain after launching "
                   + str(len(newly_launched)))

        # poll for completed jobs and store means
        newly_finished = []
        for syn_idx in remaining_jobs:
            try:
                pf = outf + "." + str(syn_idx) + ".conn"
                if ha_conn_penalty is not None:
                    pf += str(ha_conn_penalty)
                with open(pf, 'rb') as f:
                    row = pickle.load(f)
                for syn_jdx in row:
                    conn[syn_idx][syn_jdx] = row[syn_jdx]
                    # the agg algorithm makes this matrix symmetric as part of pre-processing;
                    # storing half as zeros might be cheaper
                    # conn[syn_jdx][syn_idx] = row[syn_jdx]
                newly_finished.append(syn_idx)
                os.system("rm " + pf)
                os.system("rm " + pf + ".log")
                os.system("rm err." + pf.replace("/", "-") + ".log")
            except:
                pass
        remaining_jobs = [syn_idx for syn_idx in remaining_jobs if syn_idx not in newly_finished]
        if len(newly_finished) > 0:
            print ("......... " + str(len(remaining_jobs)) + " jobs remain after gathering " +
                   str(len(newly_finished)))

        if len(remaining_jobs) >= _CONDOR_MAX_JOBS or len(newly_launched) == 0:
            time.sleep(60)

    # clean up input files
    os.system("rm " + synsets_fn)

    return conn


def main():

    # read clustering types
    polysemy_type = FLAGS_polysemy
    synonymy_type = FLAGS_synonymy
    try:
        margin = FLAGS_margin
    except NameError:
        margin = None
    distributed = True if FLAGS_distributed else False

    # read infiles
    print "reading in graph and observations..."
    f = open(FLAGS_graph_infile, 'rb')
    _, _, nps, _ = pickle.load(f)
    f.close()
    f = open(FLAGS_wnid_urls, 'rb')
    wnid_urls = pickle.load(f)
    wnids = wnid_urls.keys()
    f.close()
    if not distributed:
        f = open(FLAGS_wnid_obs_infile, 'rb')
        wnid_imgfs = pickle.load(f)
        f.close()
        f = open(FLAGS_wnid_textf_infile, 'rb')
        wnid_textfs = pickle.load(f)
        f.close()
    else:
        max_floats = 125000000 * 1  # allow ~1GB of floats in RAM for each map
        wnid_imgfs = DiskDictionary(FLAGS_wnid_obs_infile, max_floats, wnids)
        wnid_textfs = DiskDictionary(FLAGS_wnid_textf_infile, max_floats, wnids)
    f = open(FLAGS_np_obs_infile, 'rb')
    np_observations = pickle.load(f)
    f.close()
    print "... done; read "+str(len(np_observations))+" nps associated with observations out of " + \
          str(len(nps))+" total nps; wnid urls " + str(len(wnid_urls))

    # instantiate synsets containing one np for each np
    # maintain syn_obs structure which maps from (np_idx, syn_idx) tuples to list
    # of (wnid, obs_idx) tuples representing observations
    # for that np in that synset, with the latter indexing into wnid_observations
    try:
        print "instantiating synsets given previous polysemy pickle"
        pol_fn = FLAGS_polysemy_base_infile
        f = open(pol_fn, 'rb')
        synsets, syn_obs = pickle.load(f)
        f.close()
        polysemy_type = "none"
        print "... done"
    except (NameError, TypeError):
        print "instantiating synsets given observations..."
        synsets = []
        syn_obs = {}
        for np_idx in np_observations:
            if len(np_observations[np_idx]) > 0:
                synsets.append([np_idx])
                key = (np_idx, len(synsets)-1)
                syn_obs[key] = np_observations[np_idx]
        print "... done"
    # print synsets  # DEBUG
    # print syn_obs  # DEBUG

    # calculate the redundancy of feature types for early fusion based on input weighting
    imgf_weight = FLAGS_proportion_imgf_versus_textf
    imgf_fv_size = len(wnid_imgfs[wnids[0]][0])
    textf_fv_size = len(wnid_textfs[wnids[0]][0])
    if imgf_weight == 0:
        imgf_red = 0
        textf_red = 1
    elif imgf_weight == 1:
        imgf_red = 1
        textf_red = 0
    elif imgf_fv_size > textf_fv_size:
        imgf_red = 1
        textf_red = int(((imgf_fv_size * (1 - imgf_weight)) / (imgf_weight * textf_fv_size)) + 0.5)
    elif imgf_fv_size < textf_fv_size:
        imgf_red = int(((imgf_weight * textf_fv_size) / (imgf_fv_size * (1 - imgf_weight))) + 0.5)
        textf_red = 1
    else:
        imgf_red = textf_red = 1
    if textf_red > 1:  # DEBUG for weighted cosine
        textf_red = 1  # DEBUG for weighted cosine
    print ("to achieve an imgf weight of " + str(imgf_weight) + ", calculated imgf redundancy " +
           str(imgf_red) + " (" + str(imgf_fv_size) + " features) with textf redundancy " + str(textf_red) +
           " (" + str(textf_fv_size) + " features)")

    # perform polysemy detection across all synsets, splitting observation sets to create new synsets when detected
    if polysemy_type != "none":
        print "detecting polysemy across "+str(len(synsets))+" synsets ..."
        if polysemy_type != "cosine":
            if polysemy_type == "gap":
                script = "run_get_k_by_gap_statistic.py"
            elif polysemy_type == 'dpgmm':
                script = "run_get_k_by_dpgmm.py"
            elif polysemy_type == 'ms':
                script = "run_get_k_by_meanshift.py"
            elif polysemy_type == 'sc':
                script = "run_get_k_by_sc.py"
            else:
                sys.exit("ERROR: unrecognized polysemy type '" + polysemy_type + "'")
            synsets_to_remove = []
            synsets_to_add = []
            syn_obs_to_add = {}
            obs_keys_sets = []
            print "... launching '"+script+"' jobs"
            est_num_clusters = max_polysemy_dev
            unfinished_gap_scripts = []
            for syn_idx in range(0, len(synsets)):
                gap_fn = FLAGS_outfile+str(syn_idx)+".polysemy.gap_statistic.pickle"
                np_idx = synsets[syn_idx][0]  # guaranteed only one noun phrase per synset at this stage
                obs_keys = [(wnid, obs_idx) for wnid, obs_idx in syn_obs[(np_idx, syn_idx)]]
                obs_keys_sets.append(obs_keys)

                # if pickle already exists, don't launch a new job for it, but do collect it later
                try:
                    pf = open(gap_fn, 'rb')
                    _, _ = pickle.load(pf)
                    pf.close()
                    unfinished_gap_scripts.append(syn_idx)
                    continue
                except (IOError, EOFError):  # pickle hasn't been written
                    pass

                imgf_red_init = textf_red_init = 0
                if imgf_red > 0:
                    obs = [wnid_imgfs[entry[0]][entry[1]]
                           for entry in syn_obs[(np_idx, syn_idx)]]
                    imgf_red_init = 1
                else:
                    obs = [wnid_textfs[entry[0]][entry[1]]
                           for entry in syn_obs[(np_idx, syn_idx)]]
                    textf_red_init = 1
                for _ in range(imgf_red_init, imgf_red):
                    for entry_idx in range(0, len(syn_obs[(np_idx, syn_idx)])):
                        entry = syn_obs[(np_idx, syn_idx)][entry_idx]
                        obs[entry_idx].extend(wnid_imgfs[entry[0]][entry[1]])
                for _ in range(textf_red_init, textf_red):
                    for entry_idx in range(0, len(syn_obs[(np_idx, syn_idx)])):
                        entry = syn_obs[(np_idx, syn_idx)][entry_idx]
                        obs[entry_idx].extend(wnid_textfs[entry[0]][entry[1]])
                pf_fn = FLAGS_outfile+str(syn_idx)+".polysemy.observations.pickle"
                pf = open(pf_fn, 'wb')
                pickle.dump([obs, est_num_clusters, alpha_pol_density_dev], pf)
                pf.close()
                cmd = "condorify_gpu_email python "+script+" " + \
                      "--trim_poly " + str(FLAGS_trim_poly) + " " + \
                      "--min_k 1 " + \
                      "--obs_infile "+pf_fn+" --outfile "+gap_fn+" "+gap_fn+".log"
                # print "...... running '"+cmd+"'"  # DEBUG
                os.system(cmd)
                unfinished_gap_scripts.append(syn_idx)
            print "...... done"

            while len(unfinished_gap_scripts) > 0:
                time.sleep(60)  # poll for finished scripts every 60 seconds
                newly_finished = []
                print "... polling remaining jobs"
                for syn_idx in unfinished_gap_scripts:
                    gap_fn = FLAGS_outfile+str(syn_idx)+".polysemy.gap_statistic.pickle"
                    if os.path.isfile(gap_fn):
                        try:
                            pf = open(gap_fn, 'rb')
                            num_k, n_obs_classes = pickle.load(pf)
                            pf.close()
                            newly_finished.append(syn_idx)
                        except (IOError, EOFError):  # pickle hasn't been written all the way yet
                            continue
                        # print "num_k: "+str(num_k)  # DEBUG
                        pf_fn = FLAGS_outfile+str(syn_idx)+".polysemy.observations.pickle"
                        os.system("rm "+gap_fn)
                        os.system("rm "+pf_fn)
                        os.system("rm "+gap_fn+".log")
                        os.system("rm err."+gap_fn.replace("/", "-")+".log")
                        np_idx = synsets[syn_idx][0]  # guaranteed only one noun phrase per synset at this stage
                        if num_k > 1:
                            print "... splitting synset " + str(syn_idx) + " with np '"+nps[np_idx]+"'"  # DEBUG
                            # print "...... centers: "+str(num_k)+"\n...... n_obs_classes: "+str(n_obs_classes)  # DEBUG
                            synsets_to_remove.append(syn_idx)
                            for kdx in range(0, num_k):
                                synsets_to_add.append([np_idx])
                                key = (np_idx, len(synsets_to_add)-1)
                                if key not in syn_obs_to_add:
                                    syn_obs_to_add[key] = []
                                for n_obs_idx in range(0, len(n_obs_classes)):
                                    if n_obs_classes[n_obs_idx] == kdx:
                                        syn_obs_to_add[key].append(obs_keys_sets[syn_idx][n_obs_idx])
                                if len(syn_obs_to_add[key]) == 0:
                                    del synsets_to_add[-1]
                                    del syn_obs_to_add[key]
                                else:
                                    print "...... class "+str(kdx)+" got "+str(len(syn_obs_to_add[key])) + \
                                          "  of "+str(len(obs_keys_sets[syn_idx]))+" total observations"  # DEBUG
                        else:
                            print "... leaving singular synset " + str(syn_idx) + " with np '"+nps[np_idx]+"'"  # DEBUG
                unfinished_gap_scripts = [syn_idx for syn_idx in unfinished_gap_scripts
                                          if syn_idx not in newly_finished]
                print "...... done; unfinished scripts: " + str(unfinished_gap_scripts)
            synsets, _, syn_obs = update_synset_structures(nps,
                                                           synsets, synsets_to_remove, synsets_to_add,
                                                           syn_obs, syn_obs_to_add)
        else:  # use cosine splitting with margin heuristic to iteratively split synsets
            added_new_synsets = True
            synsets_to_remove = []
            synsets_to_add = []
            syn_obs_to_add = {}
            last_syn_idx = 0
            while added_new_synsets:
                added_new_synsets = False
                for syn_idx in range(last_syn_idx, len(synsets)):
                    np_idx = synsets[syn_idx][0]  # guaranteed only one noun phrase per synset at this stage
                    imgf_red_init = textf_red_init = 0
                    if imgf_red > 0:
                        obs = [wnid_imgfs[entry[0]][entry[1]]
                               for entry in syn_obs[(np_idx, syn_idx)]]
                        imgf_red_init = 1
                    else:
                        obs = [wnid_textfs[entry[0]][entry[1]]
                               for entry in syn_obs[(np_idx, syn_idx)]]
                        textf_red_init = 1
                    for _ in range(imgf_red_init, imgf_red):
                        for entry_idx in range(0, len(syn_obs[(np_idx, syn_idx)])):
                            entry = syn_obs[(np_idx, syn_idx)][entry_idx]
                            obs[entry_idx].extend(wnid_imgfs[entry[0]][entry[1]])
                    for _ in range(textf_red_init, textf_red):
                        for entry_idx in range(0, len(syn_obs[(np_idx, syn_idx)])):
                            entry = syn_obs[(np_idx, syn_idx)][entry_idx]
                            obs[entry_idx].extend(wnid_textfs[entry[0]][entry[1]])
                    obs_keys = [(wnid, obs_idx) for wnid, obs_idx in syn_obs[(np_idx, syn_idx)]]
                    n_obs = numpy.asarray(obs)
                    centers, n_obs_classes = sci_cluster.kmeans2(n_obs, 2, minit='points')
                    if len(centers) < 2:
                        # totally possible with bad initializations or aberrant data;
                        # synset isn't self-dissimilar enough if this happens
                        continue
                    sim = cosine(centers[0], centers[1])
                    # these clusters are dissimilar enough not be in the same synset
                    if sim < margin and min(n_obs_classes) != max(n_obs_classes):
                        print "... splitting synset with np '"+nps[synsets[syn_idx][0]]+"'"  # DEBUG
                        print "... centers: "+str(centers)+"\n...... n_obs_classes: " + \
                              str(n_obs_classes)+"\n...... sim: "+str(sim)  # DEBUG
                        added_new_synsets = True
                        synsets_to_remove.append(syn_idx)
                        n_obs_classes, _ = sci_cluster.vq(n_obs, centers)
                        for kdx in range(0, 2):
                            synsets_to_add.append([np_idx])
                            key = (np_idx, len(synsets_to_add)-1)
                            if key not in syn_obs_to_add:
                                syn_obs_to_add[key] = []
                            for n_obs_idx in range(0, len(n_obs_classes)):
                                if n_obs_classes[n_obs_idx] == kdx:
                                    syn_obs_to_add[key].append(obs_keys[n_obs_idx])
                            if len(syn_obs_to_add[key]) == 0:
                                del synsets_to_add[-1]
                                del syn_obs_to_add[key]
                            else:
                                print "...... class "+str(kdx)+" got "+str(len(syn_obs_to_add[key])) + \
                                      "  of "+str(len(n_obs))+" total observations"  # DEBUG
                # print "synsets_to_remove: "+str(synsets_to_remove)  # DEBUG
                # print "synsets_to_add: "+str(synsets_to_add)  # DEBUG
                # print "syn_obs_to_add: "+str(syn_obs_to_add)  # DEBUG
                synsets, last_syn_idx, syn_obs = update_synset_structures(nps,
                                                                          synsets, synsets_to_remove, synsets_to_add,
                                                                          syn_obs, syn_obs_to_add)
                synsets_to_remove = []
                synsets_to_add = []
                syn_obs_to_add = {}
                # print synsets  # DEBUG
                # print syn_obs  # DEBUG
        print "... done; expanded to "+str(len(synsets))+" synsets"
        # print synsets  # DEBUG
        # print syn_obs  # DEBUG

    # perform synonymy detection across all synsets, collapsing them and their observations when detected
    if synonymy_type != "none":
        print "detecting synonymy across "+str(len(synsets))+" synsets..."

        if synonymy_type == 'ha' or synonymy_type == 'ha_fixed' or synonymy_type == 'ha_fixed_kl':
            print "... using ha methodologies"

            # in/out filenames
            conn_fn = FLAGS_polysemy_base_infile + ".conn"
            means_fn = FLAGS_polysemy_base_infile + ".means"
            vars_fn = FLAGS_polysemy_base_infile + ".vars"
            out_fn = FLAGS_outfile + ".synonymy"
            if ".gold_" or ".none_" in FLAGS_polysemy_base_infile:  # common base has different means/connections
                suff = "." + str(imgf_red) + "." + str(textf_red)
                conn_fn += suff
                means_fn += suff
                vars_fn += suff

            # get means
            try:
                with open(means_fn, 'rb') as f:
                    _ = pickle.load(f)
                print "... loaded means from file"
                if synonymy_type == 'ha_fixed_kl':
                    with open(vars_fn, 'rb') as f:
                        _ = pickle.load(f)
                    print "... loaded vars from file"
            except:  # not written

                # get synset averages to act as actual observations for clustering
                print "... calculating means and vars across " + str(len(synsets)) + " synsets"
                means, vars = get_synset_means(FLAGS_outfile, FLAGS_wnid_urls, synsets, syn_obs,
                                               FLAGS_wnid_obs_infile, FLAGS_wnid_textf_infile,
                                               imgf_red, textf_red, imgf_fv_size, textf_fv_size)
                print "...... done; got " + str(len(means)) + " means and " + str(len(vars)) + " vars"

                print "... writing means input file for clustering algorithm"
                with open(means_fn, 'wb') as f:
                    pickle.dump(means, f)
                print "...... done"
                if synonymy_type == 'ha_fixed_kl':

                    print "... completing variance vectors by replacing zero entries with averages"
                    vars = complete_synset_vars(synsets, syn_obs, vars)
                    print "...... done"

                    print "... writing vars input file for connectivity algorithm"
                    with open(vars_fn, 'wb') as f:
                        pickle.dump(vars, f)
                    print "...... done"

            # create connectivity matrix
            try:
                ha_conn_penalty = FLAGS_ha_conn_penalty
                conn_fn += str(ha_conn_penalty)
            except:
                ha_conn_penalty = None
            try:
                ref_sets = FLAGS_ha_ref_sets
            except:
                ref_sets = -1
            if ref_sets is None:
                ref_sets = -1
            try:
                max_size = FLAGS_ha_max_size
            except:
                max_size = -1
            if max_size is None:
                max_size = -1
            if synonymy_type == 'ha_fixed_kl':
                conn_fn += "kl"
                vars_fn_or_none = vars_fn
                print "using kl divergence distance criterion for connectivity"
            else:
                vars_fn_or_none = None
            try:
                with open(conn_fn, 'rb') as f:
                    _ = pickle.load(f)
                print "... loaded connecitvity matrix from file"
            except:  # not written

                print "... calculating connectivity matrix across " + str(len(synsets)) + " synsets"
                if ha_conn_penalty is not None:
                    print "...... using conn penalty " + str(ha_conn_penalty)
                    conn = get_mean_connectivity_matrix(FLAGS_outfile, means_fn, synsets,
                                                        ha_conn_penalty=ha_conn_penalty,
                                                        vars_fn=vars_fn_or_none)
                else:
                    conn = get_mean_connectivity_matrix(FLAGS_outfile, means_fn, synsets,
                                                        vars_fn=vars_fn_or_none)
                print "...... done"

                print "... writing connectivity input file for clustering algorithm"
                with open(conn_fn, 'wb') as f:
                    pickle.dump(conn, f)
                print "...... done"

            # launch job
            print "... launching clustering job"
            est_k = int((len(synsets) / word_senses_per_cluster_dev) + 0.5)
            cmd = ("condorify_gpu_email_largemem python run_get_k_by_agg_clustering.py " +
                   "--obs_infile " + means_fn + " " +
                   "--conn_infile " + conn_fn + " " +
                   "--ref_sets " + str(ref_sets) + " ")
            if synonymy_type == 'ha_fixed' or synonymy_type == 'ha_fixed_kl':
                cmd += "--fixed_k_val " + str(est_k) + " "
            else:
                cmd += "--start_k_val " + str(est_k) + " "
            if max_size != -1:
                cmd += "--max_size " + str(max_size) + " "
            if synonymy_type == 'ha_fixed_kl':
                cmd += "--vars_infile " + str(vars_fn) + " "
            cmd += ("--outfile " + out_fn + " " +
                    out_fn + ".log")
            print "...... cmd: " + str(cmd)
            os.system(cmd)
            print "...... done"

            # poll job and finish up
            synsets_to_remove = []
            synsets_to_add = []
            syn_obs_to_add = {}
            print "... polling job"
            while True:
                time.sleep(60)
                if os.path.isfile(out_fn):
                    try:
                        pf = open(out_fn, 'rb')
                        num_k, mean_classes_orig = pickle.load(pf)
                        pf.close()
                    except (IOError, EOFError):  # pickle hasn't been written all the way yet
                        continue
                    # print "num_k: "+str(num_k)  # DEBUG
                    # re-index mean_classes to start at 0
                    os.system("rm " + out_fn)
                    os.system("rm " + out_fn + ".log")
                    os.system("rm err." + out_fn.replace("/", "-") + ".log")
                    mean_classes_map = list(set(mean_classes_orig))
                    mean_classes = [mean_classes_map.index(mean_classes_orig[m_idx])
                                    for m_idx in range(0, len(mean_classes_orig))]
                    # re-build synsets from clustering results
                    synsets_to_remove.extend(range(0, len(synsets)))
                    for _ in range(0, num_k):
                        synsets_to_add.append([])
                    for syn_idx in range(0, len(synsets)):
                        new_np_idxs = [np_idx for np_idx in synsets[syn_idx]
                                       if np_idx
                                       not in synsets_to_add[mean_classes[syn_idx]]]
                        synsets_to_add[mean_classes[syn_idx]].extend(new_np_idxs)
                        for np_idx in new_np_idxs:
                            key = (np_idx, mean_classes[syn_idx])
                            if key not in syn_obs_to_add:
                                syn_obs_to_add[key] = []
                            syn_obs_to_add[key].extend([(wnid, obs_idx)
                                                        for wnid, obs_idx
                                                        in syn_obs[(np_idx, syn_idx)]])
                    break
            synsets, last_syn_idx, syn_obs = update_synset_structures(nps,
                                                                      synsets, synsets_to_remove, synsets_to_add,
                                                                      syn_obs, syn_obs_to_add)
            print "...... done; reduced to " + str(len(synsets)) + " synsets"

            # clean up - commented out for DEBUG to re-run after completion without recalculating
            # os.system("rm " + conn_fn)
            # os.system("rm " + means_fn)
            # os.system("rm " + out_fn)

        elif synonymy_type == "buckshot":

            # in/out filenames
            sample_obs_fn = FLAGS_outfile + ".sample_obs"
            sparse_conn_fn = FLAGS_outfile + ".sparse_obs"
            means_fn = FLAGS_outfile + ".means"
            out_fn = FLAGS_outfile + ".synonymy"

            min_k = int((len(synsets) / word_senses_per_cluster_dev) + 0.5)
            try:
                with open(sample_obs_fn, 'rb') as f:
                    _ = pickle.load(f)
                with open(sparse_conn_fn, 'rb') as f:
                    _ = pickle.load(f)
                print "... loaded samples and connectivity from file"
            except:  # not written yet

                # sample sqrt(n) observations
                n = sum([len(np_observations[np_idx]) for np_idx in np_observations])
                num_samples = int(math.sqrt(n*min_k) + 0.5)
                print "... sampling sqrt(nk)=" + str(num_samples) + " of every synset's observations"  # DEBUG
                used_entries = {}
                num_sampled = 0
                while num_sampled < num_samples:
                    syn_jdx = random.randint(0, len(synsets)-1)
                    if len(synsets[syn_jdx]) == 0:
                        continue
                    num_to_sample_from_syn = random.randint(1, min(4, num_samples-num_sampled))
                    for _ in range(0, num_to_sample_from_syn):
                        syn_np_jdx = random.randint(0, len(synsets[syn_jdx])-1)
                        np_jdx = synsets[syn_jdx][syn_np_jdx]
                        if len(syn_obs[(np_jdx, syn_jdx)]) == 0:
                            continue
                        syn_np_entry_jdx = random.randint(0, len(syn_obs[(np_jdx, syn_jdx)])-1)
                        if (syn_jdx in used_entries and syn_np_jdx in used_entries[syn_jdx] and
                                syn_np_entry_jdx in used_entries[syn_jdx][syn_np_jdx]):
                            continue
                        if syn_jdx not in used_entries:
                            used_entries[syn_jdx] = {}
                        if syn_np_jdx not in used_entries[syn_jdx]:
                            used_entries[syn_jdx][syn_np_jdx] = []
                        used_entries[syn_jdx][syn_np_jdx].append(syn_np_entry_jdx)
                        num_sampled += 1
                print ("...... done; sampled " + str(num_sampled) + " of " + str(n) + " total observations" +
                       " across " + str(len(used_entries)) + " synsets of " + str(len(synsets)) + " total")

                # create connectivity matrix for sqrt(n) observations
                print "... creating connectivity matrix among sqrt(n) samples and building vectors..."
                conn = []
                obs_keys = []  # parallel to sample_obs
                sample_obs = []  # parallel to obs_keys
                for syn_idx in used_entries:
                    keys_seen = []
                    for np_idx_idx in used_entries[syn_idx]:
                        np_idx = synsets[syn_idx][np_idx_idx]
                        for key_idx in range(0, len(used_entries[syn_idx][np_idx_idx])):
                            key = syn_obs[(np_idx, syn_idx)][used_entries[syn_idx][np_idx_idx][key_idx]]
                            if key not in obs_keys:
                                obs_keys.append(key)
                                conn.append([])

                                imgf_red_init = textf_red_init = 0
                                if imgf_red > 0:
                                    new_obs = wnid_imgfs[key[0]][key[1]]
                                    imgf_red_init = 1
                                else:
                                    new_obs = wnid_textfs[key[0]][key[1]]
                                    textf_red_init = 1
                                for _ in range(imgf_red_init, imgf_red):
                                    new_obs.extend(wnid_imgfs[key[0]][key[1]])
                                for _ in range(textf_red_init, textf_red):
                                    new_obs.extend(wnid_textfs[key[0]][key[1]])
                                sample_obs.append(new_obs)

                            obs_key_idx = obs_keys.index(key)
                            keys_seen.append(obs_key_idx)
                            for key_jdx in range(key_idx+1, len(used_entries[syn_idx][np_idx_idx])):
                                key_j = syn_obs[(np_idx, syn_idx)][used_entries[syn_idx][np_idx_idx][key_jdx]]
                                if key_j not in obs_keys:
                                    obs_keys.append(key_j)
                                    conn.append([])

                                    imgf_red_init = textf_red_init = 0
                                    if imgf_red > 0:
                                        new_obs = wnid_imgfs[key_j[0]][key_j[1]]
                                        imgf_red_init = 1
                                    else:
                                        new_obs = wnid_textfs[key_j[0]][key_j[1]]
                                        textf_red_init = 1
                                    for _ in range(imgf_red_init, imgf_red):
                                        new_obs.extend(wnid_imgfs[key_j[0]][key_j[1]])
                                    for _ in range(textf_red_init, textf_red):
                                        new_obs.extend(wnid_textfs[key_j[0]][key_j[1]])
                                    sample_obs.append(new_obs)

                                obs_key_jdx = obs_keys.index(key_j)
                                for former_key in keys_seen:
                                    conn[former_key].append(obs_key_jdx)
                                keys_seen.append(obs_key_jdx)
                print "...... done"

                print "... creating connectivity matrix to write to file"
                fully_connected_conn = numpy.zeros([len(obs_keys), len(obs_keys)])
                fully_connected_conn += sys.maxint
                for obs_key_idx in range(0, len(obs_keys)):
                    if obs_key_idx in range(0, len(conn)):
                        for obs_key_jdx in conn[obs_key_idx]:
                            fully_connected_conn[obs_key_idx][obs_key_jdx] = 1
                            fully_connected_conn[obs_key_jdx][obs_key_idx] = 1
                print "...... done; " + str(len(numpy.where(fully_connected_conn == 1)[0])) + " connections"

                print "... writing buckshot input files for clustering algorithm"
                with open(sample_obs_fn, 'wb') as f:
                    pickle.dump(sample_obs, f)
                with open(sparse_conn_fn, 'wb') as f:
                    pickle.dump(fully_connected_conn, f)
                print "...... done"

            try:
                with open(means_fn, 'rb') as f:
                    _ = pickle.load(f)
                print "... loaded means from file"
            except:  # not written

                # get synset averages to act as actual observations for clustering
                # NOTE: this could be done as a map/reduce
                print "... calculating means across " + str(len(synsets)) + " synsets"
                means = []
                clusters = {}
                for syn_idx in range(0, len(synsets)):
                    obs = []
                    for np_idx in synsets[syn_idx]:
                        imgf_red_init = text_red_init = 0
                        if imgf_red > 0:
                            new_obs = [wnid_imgfs[entry[0]][entry[1]]
                                       for entry in syn_obs[(np_idx, syn_idx)]]
                            imgf_red_init = 1
                        else:
                            new_obs = [wnid_textfs[entry[0]][entry[1]]
                                       for entry in syn_obs[(np_idx, syn_idx)]]
                            text_red_init = 1
                        for _ in range(imgf_red_init, imgf_red):
                            for entry_idx in range(0, len(syn_obs[(np_idx, syn_idx)])):
                                entry = syn_obs[(np_idx, syn_idx)][entry_idx]
                                new_obs[entry_idx].extend(wnid_imgfs[entry[0]][entry[1]])
                        for _ in range(text_red_init, textf_red):
                            for entry_idx in range(0, len(syn_obs[(np_idx, syn_idx)])):
                                entry = syn_obs[(np_idx, syn_idx)][entry_idx]
                                new_obs[entry_idx].extend(wnid_textfs[entry[0]][entry[1]])
                        obs.extend(new_obs)
                    n_obs = numpy.asarray(obs)
                    clusters[syn_idx] = n_obs
                    if len(clusters[syn_idx]) > 0:
                        try:
                            means.append(numpy.mean(clusters[syn_idx], axis=0))
                        except TypeError as e:
                            print clusters[syn_idx]
                            print len(clusters[syn_idx])
                            print e
                    else:
                        means.append(numpy.zeros(imgf_red*imgf_fv_size + textf_red*textf_fv_size))
                print "...... done"

                print "... writing means input file for clustering algorithm"
                with open(means_fn, 'wb') as f:
                    pickle.dump(means, f)
                print "...... done"

            # launch job
            print "... launching clustering job"
            cmd = ("condorify_gpu_email_largemem python run_get_k_by_gap_statistic.py " +
                   "--obs_infile " + means_fn + " " +
                   "--trim_poly 1 " +  # i.e. allow singleton senses
                   "--init_obs_infile " + sample_obs_fn + " " +
                   "--init_conn_infile " + sparse_conn_fn + " " +
                   "--min_k " + str(min_k) + " " +
                   "--outfile " + out_fn + " " +
                   out_fn + ".log")
            print "...... cmd: " + str(cmd)
            os.system(cmd)
            print "...... done"

            # poll job and finish up
            synsets_to_remove = []
            synsets_to_add = []
            syn_obs_to_add = {}
            print "... polling job"
            while True:
                time.sleep(60)
                if os.path.isfile(out_fn):
                    try:
                        pf = open(out_fn, 'rb')
                        num_k, mean_classes_orig = pickle.load(pf)
                        pf.close()
                    except (IOError, EOFError):  # pickle hasn't been written all the way yet
                        continue
                    # print "num_k: "+str(num_k)  # DEBUG
                    # re-index mean_classes to start at 0
                    mean_classes_map = list(set(mean_classes_orig))
                    mean_classes = [mean_classes_map.index(mean_classes_orig[m_idx])
                                    for m_idx in range(0, len(mean_classes_orig))]
                    # re-build synsets from clustering results
                    synsets_to_remove.extend(range(0, len(synsets)))
                    for _ in range(0, num_k):
                        synsets_to_add.append([])
                    for syn_idx in range(0, len(synsets)):
                        new_np_idxs = [np_idx for np_idx in synsets[syn_idx]
                                       if np_idx
                                       not in synsets_to_add[mean_classes[syn_idx]]]
                        synsets_to_add[mean_classes[syn_idx]].extend(new_np_idxs)
                        for np_idx in new_np_idxs:
                            key = (np_idx, mean_classes[syn_idx])
                            if key not in syn_obs_to_add:
                                syn_obs_to_add[key] = []
                            syn_obs_to_add[key].extend([(wnid, obs_idx)
                                                        for wnid, obs_idx
                                                        in syn_obs[(np_idx, syn_idx)]])
                    break
            synsets, last_syn_idx, syn_obs = update_synset_structures(nps,
                                                                      synsets, synsets_to_remove, synsets_to_add,
                                                                      syn_obs, syn_obs_to_add)
            print "...... done; reduced to " + str(len(synsets)) + " synsets"

            # clean up
            os.system("rm " + sample_obs_fn)
            os.system("rm " + sparse_conn_fn)
            os.system("rm " + means_fn)
            os.system("rm " + out_fn)

        elif True:  # never do O(2^n) comparison method implemented below
            if synonymy_type == "gap":
                script = "run_get_k_by_gap_statistic.py"
            elif synonymy_type == 'dpgmm':
                script = "run_get_k_by_dpgmm.py"
            elif synonymy_type == 'ms':
                script = "run_get_k_by_meanshift.py"
            elif synonymy_type == 'sc':
                script = "run_get_k_by_sc.py"
            else:
                script = None  # will crash

            means_fn = FLAGS_outfile + ".means"
            synsets_to_remove = []
            synsets_to_add = []
            means = []
            clusters = {}
            syn_obs_to_add = {}
            try:
                with open(means_fn, 'rb') as f:
                    _ = pickle.load(f)
                print "... loaded means from file"
            except:  # not written

                print "... calculating means"
                for syn_idx in range(0, len(synsets)):
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
                    clusters[syn_idx] = n_obs
                means.extend(reevaluate_centers(clusters, len(clusters[0][0])))
                print "...... done; got "+str(len(means))+" means for "+str(len(synsets))+" synsets"

                print "... writing means input file for clustering algorithm"
                with open(means_fn, 'wb') as f:
                    pickle.dump([means,
                                 int((float(len(means))/word_senses_per_cluster_dev) + 0.5),
                                 alpha_syn_density_dev], pf)
                print "...... done"

            print "... launching '"+script+"' job to cluster means"
            gap_fn = FLAGS_outfile+"means.synonymy.gap_statistic.pickle"
            min_k = int((len(synsets) / word_senses_per_cluster_dev) + 0.5)
            cmd = "condorify_gpu_email python "+script+" " + \
                  "--trim_poly 1 " + \
                  "--min_k " + str(min_k) + " " + \
                  "--obs_infile "+means_fn+" --outfile "+gap_fn+" "+gap_fn+".log"
            print "...... cmd= '"+cmd+"'"  # DEBUG
            os.system(cmd)
            print "...... done"
            print "... polling job"
            while True:
                time.sleep(60)
                if os.path.isfile(gap_fn):
                    try:
                        pf = open(gap_fn, 'rb')
                        num_k, mean_classes_orig = pickle.load(pf)
                        pf.close()
                        # re-index mean_classes to start at 0
                        mean_classes_map = list(set(mean_classes_orig))
                        mean_classes = [mean_classes_map.index(mean_classes_orig[m_idx])
                                        for m_idx in range(0, len(mean_classes_orig))]
                    except (IOError, EOFError):  # pickle hasn't been written all the way yet
                        continue
                    # print "num_k: "+str(num_k)  # DEBUG
                    # re-build synsets from clustering results
                    synsets_to_remove = range(0, len(synsets))
                    for kdx in range(0, num_k):
                        synsets_to_add.append([])
                    for syn_idx in range(0, len(synsets)):
                        new_np_idxs = [np_idx for np_idx in synsets[syn_idx]
                                       if np_idx
                                       not in synsets_to_add[mean_classes[syn_idx]]]
                        synsets_to_add[mean_classes[syn_idx]].extend(new_np_idxs)
                        for np_idx in new_np_idxs:
                            key = (np_idx, mean_classes[syn_idx])
                            if key not in syn_obs_to_add:
                                syn_obs_to_add[key] = []
                            syn_obs_to_add[key].extend([(wnid, obs_idx)
                                                        for wnid, obs_idx
                                                        in syn_obs[(np_idx, syn_idx)]])
                    break
            print "...... done"
            synsets, last_syn_idx, syn_obs = update_synset_structures(nps,
                                                                      synsets, synsets_to_remove, synsets_to_add,
                                                                      syn_obs, syn_obs_to_add)
        else:
            added_new_synsets = True
            synsets_to_remove = []
            synsets_to_add = []
            means = []
            clusters = {}
            syn_obs_to_add = {}
            known_unpaired = []
            while added_new_synsets:
                added_new_synsets = False
                # calculate means
                print "... calculating means"
                new_clusters = {}
                for syn_idx in range(len(means), len(synsets)):
                    obs = []
                    for np_idx in synsets[syn_idx]:
                        new_obs = [wnid_imgfs[entry[0]][entry[1]]
                               for entry in syn_obs[(np_idx, syn_idx)]]
                        for _ in range(1, imgf_red):
                            for entry_idx in range(0, len(syn_obs[(np_idx, syn_idx)])):
                                entry = syn_obs[(np_idx, syn_idx)][entry_idx]
                                new_obs[entry_idx].extend(wnid_imgfs[entry[0]][entry[1]])
                        for _ in range(0, textf_red):
                            for entry_idx in range(0, len(syn_obs[(np_idx, syn_idx)])):
                                entry = syn_obs[(np_idx, syn_idx)][entry_idx]
                                new_obs[entry_idx].extend(wnid_textfs[entry[0]][entry[1]])
                        obs.extend(new_obs)
                    n_obs = numpy.asarray(obs)
                    clusters[syn_idx] = n_obs
                    new_clusters[syn_idx] = n_obs
                means.extend(reevaluate_centers(new_clusters, len(clusters[0][0])))
                print "...... done; got "+str(len(means))+" means for "+str(len(synsets))+" synsets"
                if synonymy_type == "gap":
                    # for each pair, see whether gap statistic would yield k=1 or k=2
                    print "... finding gap statistic between synsets"
                    synset_pairs_to_merge = {}  # indexed by synset idx, jdx (idx < jdx); value is gap distance
                    for syn_idx in range(0, len(synsets)):
                        if syn_idx in [key[0] for key in synset_pairs_to_merge]:  # loaded these calculations from disk
                            continue
                        print "...... launching run_gap_statistic jobs for syn_idx="+str(syn_idx)
                        unfinished_gap_scripts = []
                        for syn_jdx in range(syn_idx+1, len(synsets)):
                            if (syn_idx, syn_jdx) in known_unpaired:
                                continue
                            sim = cosine(means[syn_idx], means[syn_jdx])
                            if sim > margin:  # heuristic to save computation time
                                unfinished_gap_scripts.append(syn_jdx)
                                sub_clusters = {syn_idx: clusters[syn_idx], syn_jdx: clusters[syn_jdx]}
                                sub_means = {syn_idx: means[syn_idx], syn_jdx: means[syn_jdx]}
                                pf_fn = FLAGS_outfile+str(syn_idx)+"-"+str(syn_jdx)+".synonymy.params.pickle"
                                pf = open(pf_fn, 'wb')
                                d = [sub_clusters, sub_means, syn_idx, syn_jdx]
                                pickle.dump(d, pf)
                                pf.close()
                                gap_fn = FLAGS_outfile+str(syn_idx)+"-"+str(syn_jdx)+".synonymy.gap_statistic.pickle"
                                cmd = "condorify_gpu_email python run_gap_statistic.py " + \
                                      "--trim_poly 1 " + \
                                      "--params_infile "+pf_fn+" --outfile "+gap_fn+" "+gap_fn+".log"
                                # print "......... running '"+cmd+"'"  # DEBUG
                                os.system(cmd)
                        print "......... done"
                        print "...... polling remaining jobs"
                        while len(unfinished_gap_scripts) > 0:
                            time.sleep(10)  # poll for finished scripts every 10 seconds
                            newly_finished = []
                            for syn_jdx in unfinished_gap_scripts:
                                gap_fn = FLAGS_outfile+str(syn_idx)+"-"+str(syn_jdx)+".synonymy.gap_statistic.pickle"
                                if os.path.isfile(gap_fn):
                                    try:
                                        pf = open(gap_fn, 'rb')
                                        merge, gap = pickle.load(pf)
                                        pf.close()
                                        newly_finished.append(syn_jdx)
                                    except (IOError, EOFError):  # pickle hasn't been written all the way yet
                                        continue
                                    pf_fn = FLAGS_outfile+str(syn_idx)+"-"+str(syn_jdx)+".synonymy.params.pickle"
                                    os.system("rm "+gap_fn)
                                    os.system("rm "+pf_fn)
                                    os.system("rm "+gap_fn+".log")
                                    os.system("rm err."+gap_fn.replace("/", "-")+".log")
                                    if merge:
                                        synset_pairs_to_merge[(syn_idx, syn_jdx)] = gap
                                    else:
                                        known_unpaired.append((syn_idx, syn_jdx))
                            print "......... done; processed "+str(len(newly_finished))+" finished jobs"
                            unfinished_gap_scripts = [entry for entry in unfinished_gap_scripts
                                                      if entry not in newly_finished]
                    print "...... done"
                else:
                    # given means, get set of pairs of synsets that could be merged
                    print "... finding distances between pairs"
                    synset_pairs_to_merge = {}  # indexed by synset idx, jdx (idx<jdx); value is similarity
                    for syn_idx in range(0, len(synsets)):
                        for syn_jdx in range(syn_idx+1, len(synsets)):
                            sim = cosine(means[syn_idx], means[syn_jdx])
                            if sim > margin:
                                key = (syn_idx, syn_jdx)
                                synset_pairs_to_merge[key] = sim
                    print "...... done"
                # sort discovered merges by similarity/gap and greedily
                # merge most similar ones / ones with highest k=1 to k=2 gap
                print "... performing greedy merge of synsets"
                for key, sim in sorted(synset_pairs_to_merge.items(), key=operator.itemgetter(1), reverse=True):
                    if key[0] not in synsets_to_remove and key[1] not in synsets_to_remove:
                        print "...... merging synsets with nps "+str([nps[i] for i in synsets[key[0]]]) + \
                              " and "+str([nps[i] for i in synsets[key[1]]])  # DEBUG
                        added_new_synsets = True
                        synsets_to_remove.append(key[0])
                        synsets_to_remove.append(key[1])
                        np_idxs = synsets[key[0]][:]
                        np_idxs.extend([np_idx for np_idx in synsets[key[1]] if np_idx not in np_idxs])
                        synsets_to_add.append(np_idxs)
                        for np_idx in np_idxs:
                            syn_obs_key = (np_idx, len(synsets_to_add)-1)
                            if syn_obs_key not in syn_obs_to_add:
                                syn_obs_to_add[syn_obs_key] = []
                            for syn_idx in key:
                                if (np_idx, syn_idx) in syn_obs:
                                    syn_obs_to_add[syn_obs_key].extend([(wnid, obs_idx)
                                                                        for wnid, obs_idx
                                                                        in syn_obs[(np_idx, syn_idx)]])
                            print "......... gathered "+str(len(syn_obs_to_add[syn_obs_key]))+" observations for" + \
                                " np '"+nps[np_idx]+"' in this new synset"
                print "...... done"
                synsets, last_syn_idx, syn_obs = update_synset_structures(nps,
                                                                          synsets, synsets_to_remove, synsets_to_add,
                                                                          syn_obs, syn_obs_to_add)
                means = [means[sdx] for sdx in range(0, len(means)) if sdx not in synsets_to_remove]
                clusters_old = copy.deepcopy(clusters)
                clusters = {}
                for key in clusters_old:
                    syn_idx = key
                    for syn_jdx in synsets_to_remove:
                        if syn_jdx < key:
                            syn_idx -= 1
                    clusters[syn_idx] = clusters_old[key]
                known_unpaired_old = known_unpaired[:]
                known_unpaired = []
                for key in known_unpaired_old:
                    syn_idx = key[0]
                    syn_jdx = key[1]
                    for syn_kdx in synsets_to_remove:
                        if syn_kdx < key[0]:
                            syn_idx -= 1
                        if syn_kdx < key[1]:
                            syn_jdx -= 1
                    known_unpaired.append((syn_idx, syn_jdx))
                synsets_to_remove = []
                synsets_to_add = []
                syn_obs_to_add = {}
                # print synsets  # DEBUG
                # print syn_obs  # DEBUG
        print "... done; reduced to "+str(len(synsets))+" synsets"
        # print synsets  # DEBUG
        # print syn_obs  # DEBUG

    # write synsets, syn_obs of induced topology
    print "writing induced synsets and observation map to file..."
    f = open(FLAGS_outfile, 'wb')
    d = [synsets, syn_obs]
    pickle.dump(d, f)
    f.close()
    print "... done"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_infile', type=str, required=True,
                        help="wnid graph used to construct observations")
    parser.add_argument('--wnid_urls', type=str, required=True,
                        help="wnid urls whose idxs are used to key into distributed obs")
    parser.add_argument('--distributed', type=int, required=True,
                        help="whether observation files are distributed by wnid url idx")
    parser.add_argument('--wnid_obs_infile', type=str, required=True,
                        help="wnid observations file")
    parser.add_argument('--wnid_textf_infile', type=str, required=True,
                        help="wnid textf observations file")
    parser.add_argument('--proportion_imgf_versus_textf', type=float, required=True,
                        help="proportion in [0, 1] of weight given to image versus text features")
    parser.add_argument('--np_obs_infile', type=str, required=True,
                        help="np observations file")
    parser.add_argument('--outfile', type=str, required=True,
                        help="output pickled synsets and observations of induced graph")
    parser.add_argument('--polysemy', type=str, required=True,
                        help="polysemy detection style; 'none', 'cosine', 'gap', 'dpgmm', 'ms', 'sc'")
    parser.add_argument('--synonymy', type=str, required=True,
                        help=("synonymy detection style; 'none', 'cosine', 'gap', 'dpgmm', "
                              "'ms', 'sc', 'buckshot', 'ha', 'ha_fixed', 'ha_fixed_kl"))
    parser.add_argument('--trim_poly', type=int, required=True,
                        help="whether to trim polysemy sets to be greater than 1")
    parser.add_argument('--margin', type=float, required=False,
                        help="distance at which to split/collapse observations for 'cosine' method")
    parser.add_argument('--ha_ref_sets', type=int, required=False,
                        help="override default 100 reference sets with this value for ha synonymy")
    parser.add_argument('--ha_conn_penalty', type=int, required=False,
                        help="override default 2 connection penalty between known split senses")
    parser.add_argument('--ha_max_size', type=int, required=False,
                        help="set a maximum cluster size")
    parser.add_argument('--polysemy_base_infile', type=str, required=False,
                        help="pickle of previous polysemy operation to start from (overrides 'polysemy' arg)")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
