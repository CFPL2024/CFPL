#!/bin/bash

for ((i=0; i<5; i++))
do

python train.py \
            --dataFile "../CFDL_Data/stanford/stanford.0.spt_s_1.q_s_32.ep_200--use_schema--label_num_schema2" \
            --fileLabel "../CFDL_Data/stanford/intent.json" \
            --extendLabel "../CFDL_Data/stanford/extend_intent.json" \
            --fileModelSave "../model/ML_bert_stand_k_2_0_1shot_rand$i" \
            --studentModel "../bert-base-uncased" \
            --teacherModel "../bert-base-multilingual-uncased-massive" \
            --numDevice 0 \
            --episodeTrain 100 \
            --episodeVal 50 \
            --episodeTest 50 \
            --learning_rate 2e-5 \
            --epochs 15 \
            --numFreeze 0 \
            --warmup_steps 100 \
            --dropout_rate 0.2 \
            --weight_decay 0.1 \
            --lamda 0.1 \
            --alpha 0.8 \
            --k 2 \
            --temperature 0.05 \
            --balanced_scalar 0.9
done