# !/bin/bash

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"

# 220行 调整使用的GPU 数量
python run_nq_fine_tuning.py --path data/v10-simplified_simplified-nq-train_contextual_baseline.csv