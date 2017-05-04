import os
import sys

pols = ["none", "gap", "dpgmm", "ms", "sc"]
syns = ["none", "gap", "dpgmm", "ms", "sc"]

# for main
if False:
    folds = ["0", "1", "2", "3", "4"]
    for pol in pols:
        for syn in syns:
            for fold in folds:
                suff = pol+"_"+syn+"_"+fold
                cmd = "condorify_gpu_email python score_reconstruction.py " \
                      "--graph_infile wnid_graphs/lpob_synpol_mini.pickle " \
                      "--wnid_obs_url_infile wnid_observations/mini100.urls.pickle " \
                      "--reconstruction_infile reconstructions/mini_"+suff+".pickle " \
                      "--np_train_obs np_observations/train"+fold+".pickle " \
                      "--outfile reconstructions/mini_"+suff+".wnids.pickle " \
                      "--perf_outfile results/mini_"+suff+".graph.pickle score_"+suff+".log"
                os.system(cmd)

# for tiny
if True:
    condition = sys.argv[1]
    proportion = sys.argv[2]
    for pol in pols:
        for syn in syns:
            suff = pol+"_"+syn
            cmd = "condorify_gpu_email python score_reconstruction.py " \
                  "--graph_infile corpora/wnid_graphs/lpob_synpol_tiny.pickle " \
                  "--wnid_obs_url_infile corpora/wnid_observations/tiny100.urls-final-"+condition+".pickle " \
                  "--reconstruction_infile corpora/reconstructions/tiny100."+proportion+"-"+condition+"."+suff+".pickle " \
                  "--np_train_obs corpora/np_observations/tiny100."+condition+".train.pickle " \
                  "--outfile corpora/reconstructions/tiny100."+proportion+"-"+condition+"."+suff+".wnids.pickle " \
                  "--perf_outfile results/tiny100."+proportion+"-"+condition+"."+suff+".graph.pickle score_"+suff+".log"
            os.system(cmd)
