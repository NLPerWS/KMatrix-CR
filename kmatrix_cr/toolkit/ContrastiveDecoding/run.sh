# !/bin/bash

export CUDA_VISIBLE_DEVICES="4,5,6,7"

# pip install submitit datasets

# # Run contrastive decoding on a specified prompt:

python run_generation.py --model_name_or_path gpt2-xl --model_type gpt2 --length 256 --prompt "<|endoftext|> A version of Sonic the Hedgehog was developed by Ancient and released in 1991" --student_name_or_path gpt2 --st_coef 1.0   --student_temperature 0.5  --outfile outputs/temp_out.json    --ignore_prefix no


# # Run contrastive decoding on dataset (see submit_decoding.py for detail):
# python run_generation.py --model_name_or_path gpt2-xl --model_type gpt2 --length 256 --prompt_file wikitext --student_name_or_path gpt2 --st_coef 1.0   --student_temperature 0.5  --outfile outputs/temp_out.json    --ignore_prefix no