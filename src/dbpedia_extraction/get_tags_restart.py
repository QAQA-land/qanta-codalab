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
    parser.add_argument("-p","--page_counts", help="page counts file",required=True)
    parser.add_argument("-a","--already_done", help="Pages which are done",required=True)
    parser.add_argument("-o","--out_file", help="Output file with tags for each wikipedia page",required=True)
    args = parser.parse_args()

    # Read the page counts that are already done 
    already_done = {}
    with open(args.already_done) as f:
        for line in f:
            already_done[line.strip()] = 1
    
    #Write and checkpointing
    fw = open(args.out_file, 'a')
    fs = open(args.already_done, 'a')
    # Read the pages that are there in the training examples - page counts
    with open(args.page_counts) as f:
        for line in f:
            val = line.strip().split('\t')
            key = val[0]
            if val[0] not in already_done:
                already_done[val[0]] = 1 
                url = ''.join(["http://dbpedia.org/data/", key, '.json'])
                try:
                    data_p = requests.get(url)
                    if data_p.status_code != requests.codes.ok:
                        print ("No page found", key)
                        continue
                    data = data_p.json()
                    final_key = ''.join(["http://dbpedia.org/resource/", key])
                    if final_key not in data:
                        print ("http://dbpedia.org/resource/ not found in data", key)
                        continue
                    for k in data[final_key]:
                        if 'subject' in k:
                            val = data[final_key][k]
                            values2keep = []
                            for v in val:
                                if ':' in v['value']:
                                    values2keep.append((v['value']).split(':')[-1])
                                elif '/' in v['value']:
                                    values2keep.append((v['value']).split('/')[-1])
                            fw.write(''.join([key, '\t', 'subject', '\t', ';'.join(values2keep),'\n']))
                        elif '#type' in k:
                            val = data[final_key][k]
                            values2keep = []
                            for v in val:
                                if ':' in v['value']:
                                    values2keep.append((v['value']).split(':')[-1])
                                elif '/' in v['value']:
                                    values2keep.append((v['value']).split('/')[-1])
                            fw.write(''.join([key, '\t', 'type', '\t', ';'.join(values2keep), '\n']))
                        elif 'abstract' in k:
                            val = data[final_key][k]
                            values2keep = []
                            for v in val:
                                if 'lang' in v:
                                    if v['lang'] == 'en':
                                        values2keep.append(v['value'])
                                else:
                                    print ("not eng abstract", key)
                                    
                            fw.write(''.join([key, '\t', 'abstract', '\t', ';'.join(values2keep), '\n']))
                    fs.write(str(key)+'\n')
                    
                except requests.exceptions.RequestException as e:
                    print (e, '\t'.join([key, str(wiki[key])]) )

    fw.close()
    fs.close()