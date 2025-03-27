# !/bin/bash

export CACHE_DIR='' # just ""
export STORE_DIR=''
export MODEL_WEIGHTS_PATH=''

# base
# python3 eval_retrieve.py --mode=base --split=test --model=t5-small --cache_dir=$CACHE_DIR --store_dir=$STORE_DIR --model_weights_path=$MODEL_WEIGHTS_PATH
# ConCoRD 
python3 eval_retrieve.py --mode=gold --split=test --model=t5-small --cache_dir=$CACHE_DIR --store_dir=$STORE_DIR  --model_weights_path=$MODEL_WEIGHTS_PATH

# python-sat==1.8.dev14
# matplotlib==3.9.4
# datasets==3.3.1
# hyperopt==0.2.7