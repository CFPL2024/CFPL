#!/bin/bash

for ((i=0; i<5; i++))
do

python train.py \
            --dataFile "../CFDL_Data/crosswoz/crosswoz.0.spt_s_1.q_s_32.ep_200" \
            --fileLabel "../CFDL_Data/crosswoz/intent.json" \
            --extendLabel "../CFDL_Data/crosswoz/extend_intent.json" \
            --fileModelSave "../model/ML_bert_cross_k_2_0_1shot_rand$i" \
            --studentModel "../bert-base-chinese" \
            --teacherModel "../bert-base-multilingual-uncased-massive" \
            --numDevice 0 \
            --episodeTrain 100 \
            --episodeVal 50 \
            --episodeTest 50 \
            --learning_rate 1e-4 \
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