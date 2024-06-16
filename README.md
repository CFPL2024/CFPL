# README

Thanks for your attention. The following instructions can help you reproduce the experiments.

## Platform

Our experiments are conducted on a platform with GPU NVIDIA 3090 RTX 24GB.

## 1. prepare environment
```bash
pip install -r requirements.txt
```
## 2. prepare datasets
Full data is available, or you can generate it by your self.
Please use the generation tool for converting normal data into few-shot/meta-episode style provided by [Few-shot Data Generation Tool](https://github.com/AtmaHou/MetaDialog#few-shot-data-construction-tool)

## 3. prepare the teacher model
Download bert-base-multilingual-uncased from [huggingface](https://huggingface.co/google-bert/bert-base-multilingual-uncased).

## 4. train and test
run
```bash
bash stan_train.sh
```
to train and test models. You can get the test result in /model.


