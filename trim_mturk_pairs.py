#!/usr/bin/env python
__author__ = 'jesse'

import argparse
import pickle


# Takes in an array of mismatch maps and returns the distribution of mismatch pairs in each
# (e.g. for each synset, how many pairs of synsets did its observations end up in in comparison rec)
# as well as distribution of observations per synset -> pair maps (e.g. how many observations
# got partitioned during reconstruction)
def get_distribution(m):

    mismatches_per_synset = {}
    observations_per_mismatch = {}
    for mismatch_map in m:
        mismatches = 0
        for synset_a in mismatch_map:
            for synset_b in mismatch_map[synset_a]:
                mismatches += 1
                observations = len(mismatch_map[synset_a][synset_b])
                if observations not in observations_per_mismatch:
                    observations_per_mismatch[observations] = 0
                observations_per_mismatch[observations] += 1
        if mismatches not in mismatches_per_synset:
            mismatches_per_synset[mismatches] = 0
        mismatches_per_synset[mismatches] += 1
    return mismatches_per_synset, observations_per_mismatch


# Takes a distribution map d from instances to counts and returns an array indexed at counts
# to mass of distribution contained at or lower than each count (cumulative distribution).
def get_distribution_mass(d):

    total_mass = 0
    count_mass = {}
    for i in d:
        total_mass += d[i]*i
        if d[i] not in count_mass:
            count_mass[d[i]] = 0
        count_mass[d[i]] += d[i]*i
    cd = []
    for idx in range(0, min(count_mass.keys())):
        cd.append(0)
    for idx in range(min(count_mass.keys()), max(count_mass.keys())+1):
        cd.append(sum([count_mass[jdx] / float(total_mass) if jdx in count_mass else 0
                       for jdx in range(idx, max(count_mass.keys())+1)]))
    return total_mass, cd


# Takes in a map m of synsets -> alt_synsets -> alt_synsets -> observations.
# Returns a pair of parallel arrays, the first containing (synset, (alt_synset, alt_synset)) indices,
# the second containing the cdf of probability mass based on the observations in each pair.
# The CDF is structured such that the pair contributing the most weight is at the bottom, and the long
# tail of small observation pairs is at the top (end of list).
def get_prob_dist_based_on_observations(m, synsets=None, syn_obs_pairs=None):

    obs_pairs = []
    obs = []
    for syn_idx in range(0, len(m)):
        for alt_idx in m[syn_idx]:
            for alt_jdx in m[syn_idx][alt_idx]:
                obs_pairs.append((syn_idx, (alt_idx, alt_jdx)))
                obs.append(len(m[syn_idx][alt_idx][alt_jdx]))
                if synsets is not None:
                    obs[-1] /= float(sum([len(syn_obs_pairs[(np_idx, syn_idx)])
                                          for np_idx in synsets[syn_idx]]))
    sorted_keys = sorted(range(len(obs)), key=lambda kidx: obs[kidx], reverse=True)
    t = sum(obs)
    pairs = []
    cdf = []
    for idx in range(len(obs)-1):
        pairs.append(obs_pairs[sorted_keys[idx]])
        cdf.append(sum([obs[sorted_keys[jdx]] for jdx in range(0, idx+1)]) / float(t))
    pairs.append(obs_pairs[sorted_keys[-1]])
    cdf.append(1)
    return pairs, cdf


# Take a distribution and cdf returned by get_prob_dist_based_on_observations and return a new
# distribution and cdf in the same form but containing only the given range of the original cdf
# from (lower, upper]
def cut_dist(pairs, cdf, lower, upper):

    new_pairs = []
    masses = []
    for idx in range(0, len(pairs)):
        if lower < cdf[idx] <= upper:
            new_pairs.append(pairs[idx])
            if len(masses) == 0:
                masses.append(cdf[idx]-lower)
            else:
                masses.append(cdf[idx]-masses[-1])
    new_cdf = []
    s = float(sum(masses))
    for idx in range(0, len(new_pairs)-1):
            new_cdf.append(sum(masses[:idx+1]) / s)
    new_cdf.append(1)
    return new_pairs, new_cdf


def main():

    print "reading in hypothesized and gold reconstructions..."
    f = open(FLAGS_hyp_reconstruction_infile, 'rb')
    hyp_synsets, hyp_obs_pairs = pickle.load(f)
    f.close()
    f = open(FLAGS_gold_reconstruction_infile, 'rb')
    gold_synsets, gold_obs_pairs = pickle.load(f)
    f.close()
    print "... done"

    print "reading in mismatch maps..."
    f = open(FLAGS_maps_infile, 'rb')
    gold_maps, hyp_maps = pickle.load(f)
    f.close()
    print "... done"

    # Write observations per mismatch as probability distribution over synset pairs
    print "getting mismatches and cdfs..."
    gold_mismatch_pairs, gold_mismatch_cdf = get_prob_dist_based_on_observations(gold_maps,
                                                                                 synsets=gold_synsets,
                                                                                 syn_obs_pairs=gold_obs_pairs)
    hyp_mismatch_pairs, hyp_mismatch_cdf = get_prob_dist_based_on_observations(hyp_maps,
                                                                               synsets=hyp_synsets,
                                                                               syn_obs_pairs=hyp_obs_pairs)
    print "... done; got gold " + str(len(gold_mismatch_cdf)) + ", hyp " + str(len(hyp_mismatch_cdf))

    # Cut to specified sizes.
    print "cutting mismatch pairs to cdf range (" + str(FLAGS_cdf_lower) + " - " + str(FLAGS_cdf_upper) + "]"
    gold_trimmed_pairs, gold_trimmed_cdf = cut_dist(gold_mismatch_pairs, gold_mismatch_cdf,
                                                    FLAGS_cdf_lower, FLAGS_cdf_upper)
    hyp_trimmed_pairs, hyp_trimmed_cdf = cut_dist(hyp_mismatch_pairs, hyp_mismatch_cdf,
                                                  FLAGS_cdf_lower, FLAGS_cdf_upper)
    print "... done; cut to gold " + str(len(gold_trimmed_cdf)) + ", hyp " + str(len(hyp_trimmed_cdf))

    # Write output maps to file.
    print "writing data to file..."
    with open(FLAGS_outfile, 'wb') as f:
        d = [[gold_trimmed_pairs, gold_trimmed_cdf], [hyp_trimmed_pairs, hyp_trimmed_cdf]]
        pickle.dump(d, f)
    print "... done"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyp_reconstruction_infile', type=str, required=True,
                        help="hypothesis reconstruction structures")
    parser.add_argument('--gold_reconstruction_infile', type=str, required=True,
                        help="gold reconstruction structures")
    parser.add_argument('--maps_infile', type=str, required=True,
                        help="maps to mismatches from get_mturk_pairs.py")
    parser.add_argument('--cdf_lower', type=float, required=True,
                        help="lower bound of cdf to include in output maps [0, 1)")
    parser.add_argument('--cdf_upper', type=float, required=True,
                        help="upper bound of cdf to include in output maps (0, 1]")
    parser.add_argument('--outfile', type=str, required=True,
                        help=("pair of gold, hypothesized parallel arrays of synset idx tuples " +
                              "and cdf probs based on number of observations"))
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
