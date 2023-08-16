#!/bin/bash

# redirect stdout/stderr to a file
# exec >make-cand-test-1117.txt 2>&1

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
n_gpu=8
dir_name=$1

# modelpath
text_modelpath="../models/Text"
table_modelpath="../models/Table"
k_modelpath="../models/K"
unified_modelpath="../models/Unified"

# test
test_text_dirpath="./tmp_result/gen/${dir_name}/test/Text"
test_table_dirpath="./tmp_result/gen/${dir_name}/test/Table"
test_k_dirpath="./tmp_result/gen/${dir_name}/test/K"
test_unified_dirpath="./tmp_result/gen/${dir_name}/test/Unified"

test_text_datapath="/workspace/sangeun/dataset/NQ/Text/test.json"
test_table_datapath="/workspace/sangeun/dataset/NQ/Table/test.json"
test_k_datapath="/workspace/sangeun/dataset/NQ/K/test.json"
test_unified_datapath="/workspace/sangeun/dataset/NQ/Unified/test.json"


source ~/.bashrc
source activate fid

echo "### TEST TEXT START DATE=$(date)"

env NGPU=$n_gpu \
	python -m torch.distributed.launch --nproc_per_node=$n_gpu \
    --master_port 1234 \
	make_cand.py \
	--name make_cand_test_large \
	--model_size large \
	--eval_data $test_text_datapath \
	--eval_freq 0.5 \
	--save_freq 1.0 \
  --lr 0.00005 \
  --optim adamw \
  --scheduler fixed \
  --weight_decay 0.01 \
  --text_maxlength 200 \
	--answer_maxlength 20 \
	--model_name t5-large \
	--model_path $text_modelpath\
	--checkpoint_dir ./data/reader_checkpoint \
	--tmp_dir $test_text_dirpath\
	--accumulation_steps 1 \
  --per_gpu_batch_size 1 \
  --n_context 50 \
	--total_epoch 10 \
	--use_checkpoint

echo "### TEST TEXT END DATE=$(date)"
echo "###"


echo "### TEST TABLE START DATE=$(date)"
env NGPU=$n_gpu \
	python -m torch.distributed.launch --nproc_per_node=$n_gpu \
    --master_port 1234 \
	make_cand_beam.py \
	--name make_cand_test_large \
	--model_size large \
	--eval_data $test_table_datapath \
	--eval_freq 0.5 \
	--save_freq 1.0 \
  --lr 0.00005 \
  --optim adamw \
  --scheduler fixed \
  --weight_decay 0.01 \
  --text_maxlength 200 \
	--answer_maxlength 20 \
	--model_name t5-large \
	--model_path $table_modelpath\
	--checkpoint_dir ./data/reader_checkpoint \
	--tmp_dir $test_table_dirpath\
	--accumulation_steps 1 \
  --per_gpu_batch_size 1 \
  --n_context 50 \
	--total_epoch 10 \
	--use_checkpoint

echo "### TEST TABLE END DATE=$(date)"
echo "###"

echo "### TEST K START DATE=$(date)"
env NGPU=$n_gpu \
	python -m torch.distributed.launch --nproc_per_node=$n_gpu \
    --master_port 1234 \
	make_cand_beam.py \
	--name make_cand_test \
	--model_size large \
	--eval_data $test_k_datapath \
	--eval_freq 0.5 \
	--save_freq 1.0 \
  --lr 0.00005 \
  --optim adamw \
  --scheduler fixed \
  --weight_decay 0.01 \
  --text_maxlength 200 \
	--answer_maxlength 20 \
	--model_name t5-large \
	--model_path $k_modelpath\
	--checkpoint_dir ./data/reader_checkpoint \
	--tmp_dir $test_k_dirpath\
	--accumulation_steps 1 \
  --per_gpu_batch_size 1 \
  --n_context 50 \
	--total_epoch 10 \
	--use_checkpoint

echo "### TEST K END DATE=$(date)"
echo "###"

echo "### TEST UNI START DATE=$(date)"
env NGPU=$n_gpu \
	python -m torch.distributed.launch --nproc_per_node=$n_gpu \
    --master_port 1234 \
	make_cand_beam.py \
	--name make_cand_test \
	--model_size large \
	--eval_data $test_unified_datapath \
	--eval_freq 0.5 \
	--save_freq 1.0 \
  --lr 0.00005 \
  --optim adamw \
  --scheduler fixed \
  --weight_decay 0.01 \
  --text_maxlength 200 \
	--answer_maxlength 20 \
	--model_name t5-large \
	--model_path $unified_modelpath\
	--checkpoint_dir ./data/reader_checkpoint \
	--tmp_dir $test_unified_dirpath\
	--accumulation_steps 1 \
  --per_gpu_batch_size 1 \
  --n_context 50 \
	--total_epoch 10 \
	--use_checkpoint

echo "### TEST UNI END DATE=$(date)"
echo "###"
