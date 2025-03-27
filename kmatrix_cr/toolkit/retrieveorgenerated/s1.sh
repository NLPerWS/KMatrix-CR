# !/bin/bash

python generate.py \
    --dataset nq \
    --type Gen \
    --split none \
    --engine meta-llama/Llama-2-7b-chat-hf \
    --decoding greedy \
    --pid 1


# python find_gen_ctx_length_similar_to_ret_cxt.py \
#     --all_gen_dir Generated-context-greedy-gpt-3.5-turbo-0613/nq \
#     --ret_file backgrounds-retrieval/nq/retrieval_result.jsonl
