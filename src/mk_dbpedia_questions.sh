ABSTRACTS='dbpedia_data/abstract.json'
SUBJECTS='dbpedia_data/subject.json'
TYPE='dbpedia_data/type.json'
QB_QUESTIONS='data/qanta.mapped.2018.04.18.json'

echo $ABSTRACTS
# ABSTRACTS
python dbpedia_extraction/combine_json.py \
    --files $QB_QUESTIONS $ABSTRACTS \
    --out_file data/abstracts_questions.json

# SUBJECTS

# TYPES


