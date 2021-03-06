#!/usr/bin/env python

from __future__ import print_function


import sys
import requests
#If the module dbpediaEnquirerPy is no the python path (or same folder) you don't need to see this, this is just for this example script.
sys.path.append('../')

import argparse
import json
from dbpediaEnquirerPy import *
from collections import defaultdict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Finds tags for the wikipedia pages")
    parser.add_argument("-t","--train_file", help="training file for qanta quiz",default="/Users/tammienelson/documents/github/qanta-codalab/data/qanta.train.2018.04.18.json")
    parser.add_argument("-o","--out_file", help="Output file page counts",default="page_counts.txt")
    args = parser.parse_args()

    # Read the train examples to know the wikipedia articles
    wiki = defaultdict(int)
    question_text = {}
    with open(args.train_file, 'r') as f:
        distros_dict = json.load(f)
    questions = distros_dict['questions']
    # for i in range(0,len(questions)):
    for i in range(0,20000):
        page = questions[i]['page']
        if page not in question_text:
            question_text[page] = questions[i]
        wiki[page] += 1

    print('number of questions:',range(0,len(questions)))
    fw = open(args.out_file, "w")
    for key in wiki:
        fw.write(key + '\t' + str(wiki[key])+ '\n')
