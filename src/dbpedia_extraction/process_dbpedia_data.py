import sys
import requests
import itertools
import argparse
import string
import json
import csv
from nltk.tokenize import word_tokenize
from statistics import mean
from collections import defaultdict

def get_jaccard_sim(str1, str2):
    a = set(str1.split())
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

train_file = "/Users/tammienelson/documents/github/qanta-codalab/data/qanta.train.2018.04.18.json"
problem_pages_file = "problematic_articles"
dbpedia_file = "question_dbpedia.json"
metrics_output_file = "similarity_metrics.json"
metrics_output_file_csv = "similarity_metrics.csv"
metrics_summary_output_file = "summary_metrics.json"
metrics_summary_output_file_csv = "summary_metrics.csv"
subject_freq_output_file = "subject_freq.json"
type_freq_output_file = "type_freq.json"
subject_freq_output_file_csv = "subject_freq.csv"
type_freq_output_file_csv = "type_freq.csv"
output_for_dan_subject = "dan_data_subject.json"
output_for_dan_type = "dan_data_type.json"
output_for_dan_abstract = "dan_data_abstract.json"

custom_stopwords = ["i", "l.owl#", "wikicat", "q", "a", "an", "the", "it", "and", "or", "not", "on", "of", "to", "by", "at", "for", "from", "in", "as", "is", "``", "''"]

# read in questions and pages
wiki = defaultdict(int)
question_text = {}
problem_page_list = []
with open(train_file, 'r') as f:
    json_dict = json.load(f)
questions = json_dict['questions']
with open(dbpedia_file, 'r') as f:
    json_dict = json.load(f)
dbpedia_pages = json_dict['pages']
with open(problem_pages_file, 'r') as f:
    for line in f:
        problem_page_list.append(line.strip())

# create dict for stats
dbpedia = {}
dbpedia['metrics'] = []
dan_subject = {}
dan_subject['data'] = []
dan_type = {}
dan_type['data'] = []
dan_abstract = {}
dan_abstract['data'] = []

# create dict for type and subject frequencies
type_freq = defaultdict(int)
subj_freq = defaultdict(int)

print('question count:', len(questions))
print('answer count - excludes pages not in dbpedia:',len(dbpedia_pages))

# calculate jsim and intersection for question text compared with dbpedia data for pages -- store in dictionary
print('processing question-based metrics')
# for i in range(0,len(questions)): #use this to loop through all questions
for i in range(0,10000): # use this to restrict to subset range
    if i % 1000 == 0:
        print('iteration',i)
    question_text = questions[i]['text']
    question_text_wordlist = [w for w in word_tokenize(question_text) if not w in custom_stopwords]
    question_text = ' '.join(question_text_wordlist)
    page = questions[i]['page']
    page_check_1 = 'No page found '+ page
    page_check_2 = 'http://dbpedia.org/resource/ not found in data ' + page
    if page_check_1 in problem_page_list or page_check_2 in problem_page_list:
        print("cannot process", page)
    else:
        dbpedia_entry = (next(item for item in dbpedia_pages if item.get("page") == page))
        dbpedia_subject_wordlist = [w for w in dbpedia_entry["subject"] if not w in custom_stopwords]
        dbpedia_subject = ' '.join(dbpedia_subject_wordlist)
        dbpedia_type_wordlist = [w for w in dbpedia_entry["type"] if not w in custom_stopwords]
        dbpedia_type = ' '.join(dbpedia_type_wordlist)
        dbpedia_abstract = dbpedia_entry["abstract"]
        dbpedia_abstract_wordlist = [w for w in word_tokenize(dbpedia_abstract) if not w in custom_stopwords and not w in string.punctuation]
        # print(dbpedia_subject)
        # print(dbpedia_subject_wordlist)
        # print(dbpedia_type)
        # print(dbpedia_type_wordlist)
        # print(dbpedia_abstract)
        # print(dbpedia_abstract_wordlist)
        # calculate jaccard similarity
        jsim_subject = get_jaccard_sim(question_text, dbpedia_subject)
        jsim_type = get_jaccard_sim(question_text, dbpedia_type)
        jsim_abstract = get_jaccard_sim(question_text, dbpedia_abstract)

        # cacluate intersection
        a = dbpedia_subject.split()
        b = question_text.split()
        intersection_abstract_subject = list(set(a).intersection(set(b)))
        a = dbpedia_type.split()
        intersection_abstract_type = list(set(a).intersection(set(b)))
        a = dbpedia_abstract.split()
        intersection_abstract_abstract = list(set(a).intersection(set(b)))
        if len(dbpedia_entry["subject"]) == 0:
            subj_match_ratio = 0
        else:
            subj_match_ratio = round(len(intersection_abstract_subject)/len(dbpedia_entry["subject"]),4)
        if len(dbpedia_entry["type"]) == 0:
            type_match_ratio = 0
        else:
            type_match_ratio = round(len(intersection_abstract_type)/len(dbpedia_entry["type"]),4)
        if len(word_tokenize(dbpedia_abstract)) == 0:
            abst_match_ratio = 0
        else:
            abst_match_ratio = round(len(intersection_abstract_abstract)/len(word_tokenize(dbpedia_abstract)),4)

        # calculate frequency of type
        for dbpedia_subject in dbpedia_entry["subject_raw"]:
            dbpedia_subject = dbpedia_subject.replace (",", "")
            subj_freq[dbpedia_subject] += 1

        # calculate frequncy of subject
        for dbpedia_type in dbpedia_entry["type_raw"]:
            dbpedia_type = dbpedia_type.replace (",", "")
            type_freq[dbpedia_type] += 1

        # metrics per page
        dbpedia['metrics'].append({
            'page': page,
            # 'question_text': question_text,
            'jsim_subj': round(jsim_subject,4),
            'jsim_type': round(jsim_type,4),
            'jsim_abst':round(jsim_abstract,4),
            'subj_len': len(dbpedia_entry["subject"]),
            'subj_match_len': len(intersection_abstract_subject),
            'subj_match_ratio': subj_match_ratio,
            'type_len': len(dbpedia_entry["type"]),
            'type_match_len': len(intersection_abstract_type),
            'type_match_ratio': type_match_ratio,
            'abst_len': len(word_tokenize(dbpedia_abstract)),
            'abst_match_len': len(intersection_abstract_abstract),
            'abst_match_ratio': abst_match_ratio,
            'subject': dbpedia_entry["subject"],
            'subject_match': intersection_abstract_subject,
            'type': dbpedia_entry["type"],
            'type_match':intersection_abstract_type,
            'abstract': dbpedia_entry["abstract"],
            'abstract_match':intersection_abstract_abstract
            })

# summary metrics
max_jsim_subject = max(d['jsim_subj'] for d in dbpedia['metrics'])
min_jsim_subject = min(d['jsim_subj'] for d in dbpedia['metrics'])
mean_jsim_subject = sum(d['jsim_subj'] for d in dbpedia['metrics']) / len(dbpedia['metrics'])
max_jsim_type = max(d['jsim_type'] for d in dbpedia['metrics'])
min_jsim_type = min(d['jsim_type'] for d in dbpedia['metrics'])
mean_jsim_type = sum(d['jsim_type'] for d in dbpedia['metrics']) / len(dbpedia['metrics'])
max_jsim_abstract = max(d['jsim_abst'] for d in dbpedia['metrics'])
min_jsim_abstract = min(d['jsim_abst'] for d in dbpedia['metrics'])
mean_jsim_abstract = sum(d['jsim_abst'] for d in dbpedia['metrics']) / len(dbpedia['metrics'])
max_intersection_subject_len = max(d['subj_match_len'] for d in dbpedia['metrics'])
min_intersection_subject_len = min(d['subj_match_len'] for d in dbpedia['metrics'])
mean_intersection_subject_len = sum(d['subj_match_len'] for d in dbpedia['metrics']) / len(dbpedia['metrics'])
max_intersection_type_len = max(d['type_match_len'] for d in dbpedia['metrics'])
min_intersection_type_len = min(d['type_match_len'] for d in dbpedia['metrics'])
mean_intersection_type_len = sum(d['type_match_len'] for d in dbpedia['metrics']) / len(dbpedia['metrics'])
max_intersection_abstract_len = max(d['abst_match_len'] for d in dbpedia['metrics'])
min_intersection_abstract_len = min(d['abst_match_len'] for d in dbpedia['metrics'])
mean_intersection_abstract_len = sum(d['abst_match_len'] for d in dbpedia['metrics']) / len(dbpedia['metrics'])

summary_dbpedia_stats = {
		"max_jsim_subject" : round(max_jsim_subject,4),
        "mean_jsim_subject": round(mean_jsim_subject,4),
		"max_jsim_type" : round(max_jsim_type,4),
        "mean_jsim_type": round(mean_jsim_type,4),
		"max_jsim_abstract" : round(max_jsim_abstract,4),
        "mean_jsim_abstract": round(mean_jsim_abstract,4),
		"max_intersection_subject_len" : round(max_intersection_subject_len,4),
        "mean_intersection_subject_len": round(mean_intersection_subject_len,4),
		"max_intersection_type_len" : round(max_intersection_type_len,4),
        "mean_intersection_type_len": round(mean_intersection_type_len,4),
		"max_intersection_abstract_len" : round(max_intersection_abstract_len,4),
        "mean_intersection_abstract_len": round(mean_intersection_abstract_len,4)
	}

# loop through answers
print('processing answers data to create input files for models')
# for i in range(0,len(dbpedia_pages)): #use this to loop through all pages
for i in range(0,1): # use this to restrict to subset range
    if i % 1000 == 0:
        print('iteration',i)
    dbpedia_entry = dbpedia_pages[i]
    # output for DAN
    # print(dbpedia_entry["subject"].split(' '))
    page = dbpedia_entry["page"]
    subject_BOW = [subject_string.split(' ') for subject_string in dbpedia_entry["subject"]]
    subject_BOW = list(itertools.chain.from_iterable(subject_BOW))
    subject_BOW =  [subject_word for subject_word in subject_BOW if not subject_word.lower() in custom_stopwords]
    type_BOW = [type_string.split(' ') for type_string in dbpedia_entry["type"]]
    type_BOW = list(itertools.chain.from_iterable(type_BOW))
    type_BOW =  [type_word for type_word in type_BOW if not type_word.lower() in custom_stopwords]
    abstract_BOW = [w for w in word_tokenize(dbpedia_entry["abstract"]) if not w.lower() in custom_stopwords and not w in string.punctuation]
    # print(subject_BOW)
    # print(type_BOW)
    # print(abstract_BOW)
    dan_subject['data'].append({
        'answer': page,
        'text': subject_BOW
        })
    dan_type['data'].append({
        'answer': page,
        'text': type_BOW
        })
    dan_abstract['data'].append({
        'answer': page,
        'text': abstract_BOW
        })

# write to json
with open(metrics_output_file, 'w') as outfile:
    json.dump(dbpedia, outfile)
with open(metrics_summary_output_file, 'w') as outfile:
    json.dump(summary_dbpedia_stats, outfile)
with open(subject_freq_output_file, 'w') as outfile:
    json.dump(subj_freq, outfile)
with open(type_freq_output_file, 'w') as outfile:
    json.dump(type_freq, outfile)
# write subsets of dict to output files for dan
with open(output_for_dan_subject, 'w') as outfile:
    json.dump(dan_subject, outfile)
with open(output_for_dan_type, 'w') as outfile:
    json.dump(dan_type, outfile)
with open(output_for_dan_abstract, 'w') as outfile:
    json.dump(dan_abstract, outfile)

# write to csv
keys = dbpedia['metrics'][0].keys()
with open(metrics_output_file_csv, 'w') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writer.writerow(keys)
    dict_writer.writerows(dbpedia['metrics'])
with open(metrics_summary_output_file_csv, 'w') as f:
    for key in summary_dbpedia_stats.keys():
        f.write("%s,%s\n"%(key,summary_dbpedia_stats[key]))
with open(subject_freq_output_file_csv, 'w') as f:
    for key in subj_freq.keys():
        f.write("%s,%s\n"%(key,subj_freq[key]))
with open(type_freq_output_file_csv, 'w') as f:
    for key in type_freq.keys():
        f.write("%s,%s\n"%(key,type_freq[key]))
