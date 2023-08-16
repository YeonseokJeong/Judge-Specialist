#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
n_gpu=8
dir_name=$1
num_beams=$2
echo "### START DATE=$(date)"
echo "### HOSTNAME=$(hostname)"
echo "FiD with ssm"

source ~/.bashrc
source activate fid

# test
test_text_dirpath="./tmp_result/gen/${dir_name}/test/Text"
test_table_dirpath="./tmp_result/gen/${dir_name}/test/Table"
test_k_dirpath="./tmp_result/gen/${dir_name}/test/K"
test_unified_dirpath="./tmp_result/gen/${dir_name}/test/Unified"
select_dirpath="./tmp_result/select/${dir_name}"



env NGPU=$n_gpu \
	python -m torch.distributed.launch --nproc_per_node=$n_gpu \
    --master_port 1234 \
	test_model_select_beam.py \
	--name st_test_select \
	--model_size large \
	--eval_batches  $test_text_dirpath \
    --eval_batches2  $test_table_dirpath \
    --eval_batches3  $test_k_dirpath \
    --eval_batches_unified  $test_unified_dirpath \
	--eval_freq 0.5 \
	--save_freq 1.0 \
  --lr 0.00005 \
  --optim adamw \
  --scheduler fixed \
  --weight_decay 0.01 \
  --text_maxlength 200 \
	--answer_maxlength 20 \
	--model_name t5-large \
	--model_path ../models/Unified \
	--checkpoint_dir ./data/reader_checkpoint \
	--tmp_dir $select_dirpath \
	--accumulation_steps 1 \
  --per_gpu_batch_size 1 \
  --n_context 50 \
  --num_beams $num_beams \
	--total_epoch 10 \
	--use_checkpoint

echo "###"
echo "### END DATE=$(date)"



