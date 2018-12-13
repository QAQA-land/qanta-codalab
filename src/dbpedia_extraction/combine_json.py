import os
import sys
import json
import argparse

def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

def main():
    parser = argparse.ArgumentParser(description="Merges json files")
    parser.add_argument("-f","--files", help="json files space separated", nargs='*',required=True)
    parser.add_argument("-o","--out_file", help="Output file",required=True)
    args = parser.parse_args()

    files = args.files
    print(files)
    counter = 0
    main_json = None
    json_files = {}
    for i in files:
        with open(i, 'r') as f:
            json_files[counter] = json.load(f)
            # print json_files[counter].keys()
            if len(json_files[counter].keys()) > 1:
                main_json = counter
                # print main_json, "main json"
            counter += 1

    positional_arguments = ['qanta_id', 'first_sentence', 'gameplay', 'category', 'subcategory', 'tournament', 'difficulty', 'year', 'proto_id', 'qdb_id',  'dataset']
    for i in range(0, counter):
        # print main_json
        if i != main_json:
            # print "value of i", i
            new_questions = json_files[i]['data']
            for q in new_questions:
                if 'page' not in q:
                    q['page'] = q['answer']
                if 'fold' not in q:
                    q['fold'] = "guesstrain"
                question_text = ' '.join(q['text'])
                q['text'] = question_text
                indices = find(question_text, ' ')
                indices.insert(0,0)

                indices.insert(len(indices), indices[-1]+1)
                indices.insert(len(indices), len(question_text))
                tokens = []
                # print (indices)
                for ind in range(0, len(indices),2):
                    if ind+1 >= len(indices):
                        break
                    tokens.append([indices[ind], indices[ind+1]])

                    # print (ind, question_text[indices[ind]: indices[ind+1]], [indices[ind], indices[ind+1]])

                
                
                # question_text_len = 
                for arg_pos in positional_arguments:
                    if arg_pos not in q:
                        q[arg_pos] = None
                q['tokenizations'] = tokens
            json_files[main_json]['questions'].extend(new_questions)

    with open(args.out_file, 'w') as fw:
        json.dump(json_files[main_json], fw)

if __name__ == '__main__':
    main()

# with open(sys.argv[1], 'r') as f:
#   distros_dict = json.load(f)

# # value = {}
# # for j in distros_dict['questions']:
# #     if j['fold'] not in value:
# #         value[j['fold']] = 1
# #     else:
# #         value[j['fold']] += 1


# # for key in value:
# #     print key, value[key]

# print (len(distros_dict['data']), distros_dict.keys())
