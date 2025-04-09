# !/bin/bash


python evaluate.py --path "result/t5-small/t5-small_disent_qa_train_contextual_baseline_85540_b128_lr0.0001_e20_smax256.ckpt_dev_any_from_cf_full_subset_no_dec_len(1365,)_closed_book_inference.csv"
python evaluate.py --path "result/t5-small/t5-small_disent_qa_train_contextual_baseline_85540_b128_lr0.0001_e20_smax256.ckpt_dev_any_from_cf_full_subset_no_dec_len(1365,)_counterfactual_inference.csv"
python evaluate.py --path "result/t5-small/t5-small_disent_qa_train_contextual_baseline_85540_b128_lr0.0001_e20_smax256.ckpt_dev_any_from_cf_full_subset_no_dec_len(1365,)_factual_inference.csv"
python evaluate.py --path "result/t5-small/t5-small_disent_qa_train_contextual_baseline_85540_b128_lr0.0001_e20_smax256.ckpt_dev_any_from_cf_full_subset_no_dec_len(1365,)_random_context_inference.csv"