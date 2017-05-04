import numpy
import os
import pickle
import random
from hac import AgglomerativeClustering as hac
from sklearn.cluster import KMeans
from sklearn.cluster import k_means_
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial.distance import cosine
import sys
import time


# TODO: replace these with parameters when not under deadline pressure
# extremely non-general, hard-coded variables to control weighted cosine functions
imgf_size = 4096
textf_size = 256


def squared_euclidean(X, Y):
    return numpy.linalg.norm(X-Y)**2


def cosine_distance(X, Y):

    # implement weighted cosine if applicable
    if len(X) == imgf_size + textf_size:
        imgf_dist = cosine(X[:imgf_size], Y[:imgf_size])
        textf_dist = cosine(X[imgf_size:], Y[imgf_size:])
        return numpy.add(imgf_dist, textf_dist) / 2.0

    return cosine(X, Y)


def custom_cosine_distances(centers, X, Y_norm_squared=None, squared=False, X_norm_squared=None):

    # implement weighted cosine if applicable
    if len(X[0]) == imgf_size + textf_size:
        imgx = [X[idx][:imgf_size] for idx in range(0, len(X))]
        imgc = [centers[idx][:imgf_size] for idx in range(0, len(centers))]
        imgf_dists = cosine_distances(imgc, imgx)
        textx = [X[idx][imgf_size:] for idx in range(0, len(X))]
        textc = [centers[idx][imgf_size:] for idx in range(0, len(centers))]
        textf_dists = cosine_distances(textc, textx)
        return numpy.add(imgf_dists, textf_dists) / 2.0  # hard coded equal weight

    return cosine_distances(centers, X)


def collapse_small_clusters(x, c, n, min_cluster_size, dist):  # x observations, c cluster labels, n different clusters
    dist = convert_dist_arg_to_func(dist)

    clustering_valid = False
    while n > 1 and not clustering_valid:
        clustering_valid = True
        cluster_names = set(c)
        to_collapse = None
        collapse_obs = None
        for cluster in cluster_names:
            size = sum([1 if c_idx == cluster else 0 for c_idx in c])
            if size < min_cluster_size:
                clustering_valid = False
                to_collapse = cluster
                collapse_obs = [x[idx] for idx in range(0, len(x)) if c[idx] == cluster]
        if not clustering_valid:

            # find nearest centroid for each observation to collapse
            nearest_d = [None for _ in range(0, len(collapse_obs))]
            nearest_cluster = [None for _ in range(0, len(collapse_obs))]
            for other_cluster in cluster_names:
                if other_cluster != to_collapse:
                    oc_obs = [x[idx] for idx in range(0, len(x)) if c[idx] == other_cluster]
                    centroid = numpy.mean(oc_obs, axis=0)
                    d = [dist(centroid, collapse_obs[idx])
                         for idx in range(0, len(collapse_obs))]
                    for idx in range(0, len(collapse_obs)):
                        if nearest_d[idx] is None or d[idx] < nearest_d[idx]:
                            nearest_d[idx] = d[idx]
                            nearest_cluster[idx] = other_cluster

            # collapse
            targets_for_collapse = numpy.where(c == to_collapse)[0]
            for idx in range(0, len(collapse_obs)):
                c[targets_for_collapse[idx]] = nearest_cluster[idx]

            # re-index all clusters greater than the collapsed cluster
            for cidx in range(0, len(c)):
                if c[cidx] > to_collapse:
                    c[cidx] -= 1
            n = len(set(c))
    return c, n


# Implementation of Lloyd's algorithm used in calculation of gap statistic
# Code modified from: https://datasciencelab.wordpress.com/2013/12/12/clustering-with-k-means-in-python/

def cluster_points(set_x, mu, dist):
    # print "cluster_points called with |mu|="+str(len(mu))  # DEBUG
    clusters = {i[0]: [] for i in enumerate(mu)}
    for idx in range(0, len(set_x)):
        x = set_x[idx]

        bestmukey = None
        mindist = None
        zerodist = False
        # doing this weird indexing speeds things up because of how we initialize k means when k~~n (e.g. at the beginning of hac)
        for jdx in range(idx, len(mu)):
            d = dist(x.reshape(1, -1), mu[jdx].reshape(1, -1))
            if mindist is None or d < mindist:
                bestmukey = jdx
                mindist = d
                if numpy.isclose(mindist, 0):
                    zerodist = True
                    break
        if not zerodist:
            for jdx in range(0, min(len(mu), idx)):
                d = dist(x.reshape(1, -1), mu[jdx].reshape(1, -1))
                if mindist is None or d < mindist:
                    bestmukey = jdx
                    mindist = d
                    if numpy.isclose(mindist, 0):
                        break

        # bestmukey = min([(i[0], dist(x.reshape(1, -1), mu[i[0]].reshape(1, -1)))
        #                  for i in enumerate(mu)], key=lambda t: t[1])[0]

        if bestmukey not in clusters:
            clusters[bestmukey] = []
        clusters[bestmukey].append(x)
    # print "cluster_points returning |clusters|="+str(len(clusters))
    return clusters


def reevaluate_centers(clusters, num_feats):
    # print "reevaluate_centers called with clusters="+str(clusters)  # DEBUG
    newmu = []
    keys = sorted(clusters.keys())
    for key in keys:
        if len(clusters[key]) > 0:  # if this cluster has members, set mu as their mean
            newmu.append(numpy.mean(clusters[key], axis=0))
        else:  # else, choose a mean arbitrarily for this member-less cluster
            newmu.append(numpy.zeros(num_feats))
    # print "... reevaluate_centers returning newmu="+str(newmu)  # DEBUG
    return newmu


# Implementation of gap statistic
# Code modified from: https://datasciencelab.wordpress.com/2013/12/27/finding-the-k-in-k-means-clustering/


def wk(mu, clusters, dist):
    # print "wk called..."  # DEBUG
    # print "... |mu|="+str(len(mu))+"; mu: "+str(mu)  # DEBUG
    # print "... clusters.keys(): "+str(clusters.keys())+"; clusters: "+str(clusters)  # DEBUG
    num_k = len(mu)
    d = [sum([dist(mu[i].reshape(1, -1), c.reshape(1, -1)) for c in clusters[i]]) for i in range(0, num_k)]
    try:
        wk_value = sum([d[i]/(2.0*len(clusters[i])) for i in range(0, num_k)])
    except ZeroDivisionError:  # e.g. one of the clusters is empty
        wk_value = float('inf')
    # print "...returning wk="+str(wk_value)  # DEBUG
    return wk_value


def bounding_hypercube(set_x):
    # print "bounding_hypercube called"  # DEBUG
    dim_min = []
    dim_max = []
    for dim_idx in range(0, len(set_x[0])):
        dim_min.append(min(set_x, key=lambda a: a[dim_idx])[dim_idx])
        dim_max.append(max(set_x, key=lambda a: a[dim_idx])[dim_idx])
    return dim_min, dim_max


NUM_REFERENCE_SETS = 100


# Reference points have features that are bound to the min/max of that feature in set_x
def get_reference_sets(set_x):
    # print "get_refrence_sets creating " + str(NUM_REFERENCE_SETS) + " reference sets"  # DEBUG
    ref_sets = []
    dim_min, dim_max = bounding_hypercube(set_x)
    for i in range(0, NUM_REFERENCE_SETS):
        xb = []
        for n in range(0, len(set_x)):
            xb.append([random.uniform(dim_min[dim_idx], dim_max[dim_idx])
                       for dim_idx in range(0, len(dim_min))])
        xb = numpy.array(xb)
        ref_sets.append(xb)
    return ref_sets


# Reference points are independent, random unit vectors
def get_reference_sets_cosine_null(N, dim):
    # print "get_refrence_sets creating " + str(NUM_REFERENCE_SETS) + " reference sets"  # DEBUG
    ref_sets = []
    for i in range(0, NUM_REFERENCE_SETS):
        xb = numpy.array([make_rand_unit_vector(dim) for _ in range(0, N)])
        ref_sets.append(xb)
    return ref_sets


# Return a random unit vector of given dimension
def make_rand_unit_vector(dim):
    vec = [random.gauss(0, 1) for _ in range(dim)]
    mag = sum(x**2 for x in vec) ** .5
    return [x/mag for x in vec]


# returns True if lower k should be accepted by gap statistic
def gap_statistic(ks, wks, ref_sets, dist, prev=None, use_condor=False, reverse=False, pos_gap=False):
    wkbs = numpy.zeros(len(ks))
    sk = numpy.zeros(len(ks))
    if prev is not None:
        wkbs[0] = prev[0]
        sk[0] = prev[1]
    for entry in ks:
        indk, num_k = entry
        if prev is not None and indk == 0:
            continue
        # print "gap_statistic clustering reference data for num_k="+str(num_k)  # DEBUG
        # Create reference datasets (or retrieve their stats from cache)
        bwkbs = numpy.zeros(NUM_REFERENCE_SETS)
        clusterings_to_collect = {}
        for i in range(0, NUM_REFERENCE_SETS):
            xb = ref_sets[i]
            # print "...gap_statistic clustering uniform sample for num_k="+str(num_k)  # DEBUG

            if not use_condor:
                km = KMeans(n_clusters=num_k, init=xb[:num_k], n_init=1)
                km.fit(xb)
                # print "......done"  # DONE
                mu = km.cluster_centers_
                clusters = cluster_points(xb, mu, dist)
                bwkbs[i] = numpy.log(wk(mu, clusters, dist))
            else:  # using condor hard-codes cosine distance for now
                ref_fn = str(time.time())+".pickle"
                clusterings_to_collect[i] = ref_fn
                d = [num_k, xb]
                with open(ref_fn, 'wb') as f:
                    pickle.dump(d, f)
                cmd = ("condorify_gpu_email python run_k_means.py " +
                       "--infile " + ref_fn + " " +
                       "--outfile " + ref_fn + ".out " +
                       ref_fn + ".log")
                os.system(cmd)

        if use_condor:
            while len(clusterings_to_collect) > 0:
                newly_collected = []
                for i in clusterings_to_collect:
                    ref_fn = clusterings_to_collect[i]
                    try:
                        with open(ref_fn + ".out", 'rb') as f:
                            d = pickle.load(f)
                            mu = d[0]
                            clusters = d[1]
                            os.system("rm " + ref_fn)
                            os.system("rm " + ref_fn + ".out")
                            os.system("rm " + ref_fn + ".log")
                            os.system("rm err." + ref_fn + ".log")
                    except:
                        continue
                    bwkbs[i] = numpy.log(wk(mu, clusters, dist))  # reduce
                    newly_collected.append(i)

                # if len(newly_collected) > 0:
                #     print "......collected means for " + str(len(newly_collected)) + " ref sets"  # DEBUG
                for i in newly_collected:
                    del clusterings_to_collect[i]
                if len(newly_collected) == 0:
                    time.sleep(60)

        wkbs[indk] = sum(bwkbs) / NUM_REFERENCE_SETS
        sk[indk] = numpy.sqrt((sum(bwkbs - wkbs[indk]) ** 2) / NUM_REFERENCE_SETS)
        sk[indk] *= numpy.sqrt(1 + (1/NUM_REFERENCE_SETS))
        # print "bwkbs: "+str(bwkbs)  # DEBUG
    gap_prev = wkbs[0] - wks[0]
    gap_curr = wkbs[1] - wks[1]
    # print "ks: "+str(ks)  # DEBUG
    # print "wks: "+str(wks)  # DEBUG
    # print "wkbs: "+str(wkbs)  # DEBUG
    # print "sk: "+str(sk)  # DEBUG
    print "gap_prev: "+str(gap_prev)  # DEBUG
    print "gap_curr: "+str(gap_curr)  # DEBUG
    if not reverse:
        gap = gap_prev - (gap_curr - sk[1])  # - 1  # DEBUG; -1 pushes margin a little further to encourage higher k
    else:
        gap = gap_curr - (gap_prev - sk[0])
    done = (gap >= 0)
    if reverse and pos_gap:  # we are now looking for a negative gap, having already found the positive one
        done = (gap < 0)
    return done, gap, [wkbs[1], sk[1]]


def get_k_by_gap_statistic(set_x, min_cluster_size, start_k, dist='squared_euclidean', buckshot=None):
    # print "get_k_by_gap_statistic called"  # DEBUG
    dist = convert_dist_arg_to_func(dist)

    # Dispersion for real distribution
    b = start_k
    found_optimum_k = False
    l = [None, None]
    wks = [None, None]
    ref_sets = get_reference_sets(set_x)
    prev = None
    while not found_optimum_k:
        b += 1  # so b starts at 2; comparing k=1 against k=2
        if b > len(set_x) / min_cluster_size:  # gap statistic no longer relevant; choose k = |X| / min_cluster_size
            found_optimum_k = True
            continue
        ks = [[0, b-1], [1, b]]
        for entry in ks:
            indk, num_k = entry
            print "gap_statistic clustering " + str(len(set_x)) + " data for num_k=" + str(num_k)  # DEBUG
            if indk == 0 and l[1] is not None:
                w = wks[1]
                labels = l[1]
            else:

                if buckshot is not None:
                    sys.exit("ERROR: no longer support buckshot clustering")
                    print "performing buckshot initialization with num_k=" + str(num_k)  # DEBUG
                    init_x, init_conn = buckshot
                    aggc = AgglomerativeClustering(linkage='average', affinity='cosine',
                                                   connectivity=init_conn, n_clusters=num_k)
                    aggl = aggc.fit_predict(init_x)
                    aggm = [numpy.mean([init_x[idx] for idx in range(0, len(init_x))
                                        if aggl[idx] == agg_label], axis=0)
                            for agg_label in set(aggl)]
                    print "... done"  # DEBUG
                    km = KMeans(n_clusters=num_k, init=numpy.asarray(aggm), n_init=1)
                else:
                    km = KMeans(n_clusters=num_k, init='random')
                km.fit(set_x)
                mu = km.cluster_centers_
                labels = km.labels_
                clusters = cluster_points(set_x, mu, dist)
                w = numpy.log(wk(mu, clusters, dist))
            l[indk] = labels
            wks[indk] = w
        found_optimum_k, gap, prev = gap_statistic(ks, wks, ref_sets, dist, prev=prev)
    if l[0] is None:
        return None, None
    return len(set(l[0])), l[0]


def get_k_by_gap_statistic_agg(set_x, conn, n_ref_sets, start_k=None):
    # print "get_k_by_gap_statistic_agg called"  # DEBUG
    global NUM_REFERENCE_SETS
    NUM_REFERENCE_SETS = n_ref_sets  # reduce for computation time sanity

    # Dispersion for real distribution
    if start_k is None:
        start_k = len(set_x)-1
    b = start_k+1
    found_optimum_k = False
    found_pos = False  # need to find a positive gap and then continue finding positive gaps until a negative at the bottom of the slope
    l = [None, None]
    wks = [None, None]
    ref_sets = get_reference_sets(set_x)
    # ref_sets = get_reference_sets_cosine_null(len(set_x), 100)  # small-ish unit vectors
    prev = None
    agg = hac(n_clusters=1, affinity='precomputed',
              connectivity=conn, linkage='average')
    label_generator = agg.fit(set_x)
    for _ in range(0, len(set_x)-(start_k+1)):
        _ = label_generator.next()
    while not found_optimum_k:
        b -= 1  # e.g. hac must make at least one merge
        if b == 1:  # gap statistic no longer relevant; choose k = 1
            found_optimum_k = True
            continue
        ks = [[0, b], [1, b-1]]
        for entry in ks:
            indk, num_k = entry
            print "gap_statistic clustering " + str(len(set_x)) + " data for num_k=" + str(num_k)  # DEBUG
            if indk == 0 and l[1] is not None:
                w = wks[1]
                label_assignments = l[1]
            else:
                label_assignments = label_generator.next()
                labels = set(label_assignments)
                print "... got " + str(len(set(labels))) + " unique labels by hac yield"  # DEBUG
                labels_to_idxs = {l: numpy.where(label_assignments==l)[0] for l in labels}
                # print "... built labels_to_idxs structure"  # DEBUG
                if len(set(labels)) != num_k:
                    sys.exit("... ERROR: num unique labels " + str(len(set(labels))) +
                             " does not match target num_k " + str(num_k))
                clusters = {}
                mu = []
                for label in labels:
                    clusters[label] = [set_x[idx] for idx in labels_to_idxs[label]]
                    if len(labels_to_idxs[label]) > 1:
                        mu.append(numpy.mean([set_x[idx] for idx in labels_to_idxs[label]], axis=0))
                    else:
                        mu.append(set_x[labels_to_idxs[label][0]])
                print "... calculated " + str(len(mu)) + " cluster means and " + str(len(clusters)) + " clusters"  # DEBUG
                mu = numpy.asarray(mu)
                w = numpy.log(wk(mu, clusters, cosine_distances))
                print "... caclulated w score"  # DEBUG
            l[indk] = label_assignments
            wks[indk] = w
        dec, gap, prev = gap_statistic(ks, wks, ref_sets, cosine_distances,
                                       prev=prev, use_condor=False, reverse=True, pos_gap=found_pos)
        if dec:
            if found_pos:
                found_optimum_k = True
                print "new gap is negative; select higher k"  # DEBUG
            else:
                found_pos = True
                print "new gap is positive; now looking for bend to negative slope"  # DEBUG
        print "b: " + str(b) + ", gap: " + str(gap)  # DEBUG
    if l[0] is None:
        return None, None
    return len(set(l[0])), l[0]


def convert_dist_arg_to_func(dist):

    if dist == 'squared_euclidean':
        dist = squared_euclidean
    elif dist == 'cosine':
        dist = cosine_distance
        k_means_.euclidean_distances = custom_cosine_distances
    else:
        sys.exit("FATAL: unrecognized distance metric '" + dist + "'")

    return dist
