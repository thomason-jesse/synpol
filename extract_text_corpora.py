#!/usr/bin/env python
__author__ = 'jesse'
''' give this a set of np_observations

    outputs synsets and observation pairing attempting to reconstruct original wnid_graph based on observations alone

'''

import argparse
import pickle
import os
import ast
import urllib2
import socket
import httplib
import re
import subprocess
import time
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize


def visible(element):
    if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:
        return False
    elif re.match('<!--.*-->', unicode(element)):
        return False
    return True


fw = open('/usr/share/dict/words', 'r')
words_set = set([line.strip() for line in fw.readlines()])
fw.close()


def clean_and_trim(l, n):
    r = []
    for s in l:
        sl = s.lower()
        t = word_tokenize(sl)
        wt = [tk for tk in t if tk in words_set]
        if len(wt) > n:
            r.append(wt)
    return r


def main():

    # read infiles
    print "reading in observation urls..."
    f = open(FLAGS_wnid_obs_urls_infile, 'rb')
    wnid_observations_urls = pickle.load(f)
    f.close()
    unique_filepath = FLAGS_wnid_obs_urls_infile.replace("/", "--")
    print "... done"

    # read partial observations if any
    try:
        print "reading in partial text observations..."
        f = open(FLAGS_partial_texts, 'rb')
        wnid_observations_texts = pickle.load(f)
        f.close()
        print "... done"
    except TypeError:
        wnid_observations_texts = {}

    # launch server
    print "launching server"  # this will just quietly fail if there's already a server running
    launched = False
    p = None  # server
    while not launched:
        p = subprocess.Popen("python mrisa-master/server/mrisa_server.py".split())
        print "... launched server"
        launched = True
        time.sleep(10)  # let the server get up if it's gonna
        if p.poll() is not None:
            print "... couldn't get a lock on server; sleeping 30 minutes"
            launched = False
            time.sleep(60*30)  # wait half an hour and try to lock our own server again
    print "... done"

    # for each wnid, gather text for every url
    for wnid in wnid_observations_urls:
        if wnid in wnid_observations_texts:
            print "already have text for wnid "+str(wnid)  # DEBUG
            continue
        print "getting text for wnid "+str(wnid)  # DEBUG
        wnid_observations_texts[wnid] = []
        for url in wnid_observations_urls[wnid]:
            print "... getting text for instance "+str(url)  # DEBUG

            # get a list of pages from which to grab text
            # cmd = "curl -X POST -H \"Content-Type: application/json\" -d '{\"image_url\":\"" +
            #           url + "\"}' http://localhost:5000/search > resp.txt"
            search_f = "search_"+unique_filepath
            cmd = "wget --output-document " + search_f + " --post-data '{\"image_url\":\"" + \
                  url + "\"}' --header 'Content-Type: application/json' http://localhost:5000/search 2> /dev/null"
            os.system(cmd)
            try:
                f = open(search_f, 'r')
                d = ast.literal_eval(f.read())
                # get subset that aren't /search/ links from google cache or image-net domain itself
                page_urls = [l for l in d["links"] if l[0] != "/" and "image-net" not in l]
                f.close()
                os.system("rm " + search_f)
            except (IOError, SyntaxError):  # failed to pull down page or corruption in page
                print "...... FAILED to complete reverse image search"
                page_urls = []

            # grab text from retrieved list of page urls
            text = []
            for purl in page_urls:
                print "...... extracting text for page "+str(purl)  # DEBUG
                try:
                    html = urllib2.urlopen(purl, timeout=60).read()
                    soup = BeautifulSoup(html)
                    ptext = soup.findAll(text=True)
                    ptext = filter(visible, ptext)
                    snippets = clean_and_trim(ptext, 3)
                    text.extend(snippets)
                    print "......... done; got "+str(len(snippets))+" text snippets"  # DEBUG
                except (IOError, socket.timeout, httplib.BadStatusLine,
                        httplib.IncompleteRead, UnicodeDecodeError, TypeError,
                        httplib.HTTPException):
                    print "......... FAILED"  # DEBUG
            wnid_observations_texts[wnid].append(text)
            print "...... done; got "+str(len(text))+" total text snippets for image"  # DEBUG
        print "... done"  # DEBUG

    # kill server process so other jobs on this machine can begin
    p.kill()

    # write synsets, syn_obs of induced topology
    print "writing all wnids with extracted text to file..."
    f = open(FLAGS_outfile, 'wb')
    d = wnid_observations_texts
    pickle.dump(d, f)
    f.close()
    print "... done"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wnid_obs_urls_infile', type=str, required=True,
                        help="wnid observation urls file")
    parser.add_argument('--outfile', type=str, required=True,
                        help="output pickled map from synsets to instance text lists")
    parser.add_argument('--partial_texts', type=str, required=False,
                        help="existing texts for some subset of wnids")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
