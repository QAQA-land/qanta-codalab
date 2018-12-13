mkdir -p $1
python ../evaluate.py \
    --curve-pkl ../curve_pipeline.pkl \
    --wait 5 \
    --hostname 0.0.0.0 \
    --norun-web \
    --char_step_size 600 \
    --output_prefix $1 \
    ../src/data/qanta.dev.2018.04.18.json
    
