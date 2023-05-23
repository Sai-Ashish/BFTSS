#!/bin/bash

GPU=0
TASK=qqp
TRAIN_SPLIT=100
MODE=MLO_approach_topk # choose mode
MODEL=bert-large-cased
BSZ=16
EPOCH=3.0
S_EPOCHS=6
LR=5e-6
S_LR=4e-5
W_LR=4e-5
CLUSTER_DIM=50

for i in `seq 10`
do
  
  OUTPUT_DIR=outputs/bert_low_output_${i}
  SEED=$RANDOM

  CUDA_VISIBLE_DEVICES=${GPU} python run_glue_low_resource.py \
  --model_name_or_path ${MODEL} \
  --task_name ${TASK} \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size ${BSZ} \
  --learning_rate ${LR} \
  --num_train_epochs ${EPOCH} \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --output_dir ${OUTPUT_DIR} \
  --seed ${SEED} \
  --save_total_limit 1 \
  --save_steps 30000 \
  --overwrite_output_dir \
  --mode ${MODE} \
  --MLO true \
  --sample_p 1.0 \
  --inv_temp 4.0 \
  --S_init MLO\
  --train_split ${train_split} \
  --S_lr ${S_LR} \
  --W_lr ${W_LR} \
  --S_wd 0 \
  --S_epochs ${S_EPOCHS} \
  --cluster_dim ${CLUSTER_DIM}
done
