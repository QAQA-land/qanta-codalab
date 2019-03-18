#!/usr/bin/env python

from __future__ import print_function


import sys
import requests
import string
#If the module dbpediaEnquirerPy is no the python path (or same folder) you don't need to see this, this is just for this example script.
sys.path.append('../')

import argparse
import json
from dbpediaEnquirerPy import *
from collections import defaultdict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Finds tags for the wikipedia pages")
    parser.add_argument("-p","--page_counts", help="page counts file",default="page_counts.txt")
    parser.add_argument("-a","--already_done", help="Pages which are done",default="already_done.txt")
    parser.add_argument("-o","--out_file", help="Output file with tags for each wikipedia page",default="question_dbpedia.json")
    args = parser.parse_args()

    # store the dbpedia data in dictionary; prep to write as json
    dbpedia_entries = {}
    dbpedia_entries['pages'] = []

    # Read the page counts that are already done
    already_done = {}
    with open(args.already_done) as f:
        for line in f:
            already_done[line.strip()] = 1

    #Write and checkpointing
    fw = open(args.out_file, 'a')
    fs = open(args.already_done, 'a')
    # Read the pages that are there in the training examples - page counts
    num_processed = 0
    with open(args.page_counts) as f:
        for line in f:
            num_processed += 1
            if num_processed % 1000 == 0:
                print('*****processed', num_processed)
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
                        data = None
                        continue
                    dbpedia_subject = []
                    dbpedia_type = []
                    dbpedia_abstract = ""
                    for k in data[final_key]:
                        if 'subject' in k:
                            val = data[final_key][k]
                            values2keep = []
                            values2keep_raw = []
                            for v in val:
                                if str(v['value']).count(":") == 2:
                                    value2add = (v['value']).split(':')[-1]
                                elif '/' in str(v['value']):
                                    value2add = ((v['value']).split('/')[-1])
                                # split into words by replacing underscore with space
                                word_split = value2add.replace ("_", " ")
                                values2keep_raw.append(value2add)
                                values2keep.append(word_split)
                            dbpedia_subject_raw = values2keep_raw
                            dbpedia_subject = values2keep
                            # fw.write(''.join([key, '\t', 'subject', '\t', ';'.join(values2keep),'\n']))
                        elif '#type' in k:
                            val = data[final_key][k]
                            values2keep = []
                            values2keep_raw = []
                            for v in val:
                                if str(v['value']).count(":") == 2:
                                    value2add = (v['value']).split(':')[-1]
                                elif '/' in str(v['value']):
                                    value2add = ((v['value']).split('/')[-1])
                                # remove id numbers at end of string and separate camel case into words
                                # caveats: we lose years when they are at the end of the type string; loose spaces when before numbers
                                value2add = value2add.rstrip(string.digits)
                                upper_index = [i for i, e in enumerate(value2add) if e.isupper()] + [len(value2add)]
                                word_split = ' '.join([value2add[x: y] for x, y in zip(upper_index, upper_index[1:])])
                                values2keep_raw.append(value2add)
                                values2keep.append(word_split)
                            dbpedia_type_raw = values2keep_raw
                            dbpedia_type = values2keep
                            # fw.write(''.join([key, '\t', 'type', '\t', ';'.join(values2keep), '\n']))
                        elif 'abstract' in k:
                            val = data[final_key][k]
                            values2keep = []
                            for v in val:
                                if 'lang' in v:
                                    if v['lang'] == 'en':
                                        values2keep.append(v['value'])
                                else:
                                    print ("not eng abstract", key)
                            if len(values2keep) > 0:
                                dbpedia_abstract = values2keep[0]
                            else:
                                print('unable to load abstract for', key)
                            # fw.write(''.join([key, '\t', 'abstract', '\t', ';'.join(values2keep), '\n']))
                    dbpedia_entries['pages'].append({
                        'page': key,
                        'subject': dbpedia_subject,
                        'subject_raw': dbpedia_subject_raw,
                        'type':dbpedia_type,
                        'type_raw': dbpedia_type_raw,
                        'abstract':dbpedia_abstract})

                except requests.exceptions.RequestException as e:
                    print (e, '\t'.join([key, str(wiki[key])]) )

    with open(args.out_file, 'w') as outfile:
        json.dump(dbpedia_entries, outfile)

    fw.close()
    fs.close()
