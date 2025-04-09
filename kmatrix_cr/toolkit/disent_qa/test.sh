# !/bin/bash

# answer_type:  <f/cf/rc/cb>

export CUDA_VISIBLE_DEVICES="4,5,6,7"
python query_model.py --answer_type f --path ../../../kmatrix_cr_datasets/CM/disent_qa/simplified-nq-dev_any_from_cf_full_subset_no_dec.csv --checkpoint_name ../../../kmatrix_cr_models/disent_qa/t5-small_disent_qa_train_contextual_baseline_85540_b128_lr0.0001_e20_smax256.ckpt
python query_model.py --answer_type cf --path ../../../kmatrix_cr_datasets/CM/disent_qa/simplified-nq-dev_any_from_cf_full_subset_no_dec.csv --checkpoint_name ../../../kmatrix_cr_models/disent_qa/t5-small_disent_qa_train_contextual_baseline_85540_b128_lr0.0001_e20_smax256.ckpt
python query_model.py --answer_type rc --path ../../../kmatrix_cr_datasets/CM/disent_qa/simplified-nq-dev_any_from_cf_full_subset_no_dec.csv --checkpoint_name ../../../kmatrix_cr_models/disent_qa/t5-small_disent_qa_train_contextual_baseline_85540_b128_lr0.0001_e20_smax256.ckpt
python query_model.py --answer_type cb --path ../../../kmatrix_cr_datasets/CM/disent_qa/simplified-nq-dev_any_from_cf_full_subset_no_dec.csv --checkpoint_name ../../../kmatrix_cr_models/disent_qa/t5-small_disent_qa_train_contextual_baseline_85540_b128_lr0.0001_e20_smax256.ckpt