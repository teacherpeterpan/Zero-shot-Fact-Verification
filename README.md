# Zero-shot-Fact-Verification-by-Claim-Generation
This repository contains code and models for the paper: [Zero-shot Fact Verification by Claim Generation (ACL-IJCNLP 2021)](). 

- We explore the possibility of automatically **generating large-scale (evidence, claim) pairs** to train the fact verification model. 

- We propose a simple yet general framework **Question Answering for Claim Generation (QACG)** to generate three types of claims from any given evidence: 1) claims that are **supported** by the evidence, 2) claims that are **refuted** by the evidence, and 3) claims that the evidence does **Not have Enough Information (NEI)** to verify. 

- We show that the generated training data can greatly benefit the fact verification system in both **zero-shot and few-shot learning settings**. 

#### General Framework of QACG

<p align="center">
<img src=Resource/framework.png width=700/>
</p>

#### Example of Generated Claims

<p align="center">
<img src=Resource/examples.png width=700/>
</p>

## Requirements

- Python 3.7.3
- torch 1.7.1
- tqdm 4.49.0
- transformers 4.3.3
- stanza 1.1.1
- nltk 3.5
- scikit-learn 0.23.2
- sense2vec

## Data Preparation

The data used in our paper is constructed based on the original [FEVER dataset](https://fever.ai/resources.html). We use the gold evidence sentences in FEVER for the SUPPORTED and REFUTED claims. We collect evidence sentences for the NEI class using the retrival method proposed in [**the Papelo system from FEVER'2018**](https://github.com/cdmalon/fever2018-retrieval). The detailed data processing process is introduced [here](./data_processing.md). 

Our processed dataset is publicly available in the Google Cloud Storage: [https://storage.cloud.google.com/few-shot-fact-verification/](https://storage.cloud.google.com/few-shot-fact-verification/)

You could download them to the `data` folder using `gsutil`:
```shell
gsutil cp gs://few-shot-fact-verification/data/* ./data/
```

There are two files in the folder:
- `fever_train.processed.json`
- `fever_dev.processed.json`

One data sample is as follows: 

```json
{
    "id": 22846,
    "context": "Penguin Books was founded in 1935 by Sir Allen Lane as a line of the publishers The Bodley Head , only becoming a separate company the following year .",
    "ori_evidence": [
      [
        "Penguin_Books",
        1,
        "It was founded in 1935 by Sir Allen Lane as a line of the publishers The Bodley Head , only becoming a separate company the following year ."
      ]
    ],
    "claim": "Penguin Books is a publishing house founded in 1930.",
    "label": "REFUTES"
}
```

## Claim Generation

Given a piece of evidence in FEVER, we generate three different types of claims: SUPPORTED, REFUTED, and NEI. The codes are in `Claim_Generation` folder. 

### a) NER Extraction

First, we extract all Name Entities (NERs) in the evidence. 

```shell
mkdir -p ../output/intermediate/

python Extract_NERs.py \
    --train_path ../data/fever_train.processed.json \
    --dev_path ../data/fever_dev.processed.json \
    --save_path ../output/intermediate/
```

### b) Question Generation

Then, we generate (question, answer) pairs from the evidence given an named entity as the answer. 

For question generator, we use the pretrained QG model from [patil-suraj](https://github.com/patil-suraj/question_generation), a Google T5 model finetuned on the SQuAD 1.1 dataset. Given an input text *D* and an answer *A*, the question generator outputs a question *Q*. 

Run the following codes to generate (Q,A) pairs for the entire FEVER dataset. 

```shell
python Generate_QAs.py \
    --train_path ../data/fever_train.processed.json \
    --dev_path ../data/fever_dev.processed.json \
    --data_split train \
    --entity_dict ../output/intermediate/entity_dict_train.json \
    --save_path ../output/intermediate/precompute_QAs_train.json

python Generate_QAs.py \
    --train_path ../data/fever_train.processed.json \
    --dev_path ../data/fever_dev.processed.json \
    --data_split dev \
    --entity_dict ../output/intermediate/entity_dict_dev.json \
    --save_path ../output/intermediate/precompute_QAs_dev.json
```

### c) Claim Generation

We use the pretrained [Sense2Vec (Trask et. al, 2015)](https://github.com/explosion/sense2vec) to find answer replacements for generating REFUTED claims. The pretrained model can be downloaded [here](https://github.com/explosion/sense2vec/releases/download/v1.0.0/s2v_reddit_2015_md.tar.gz). Download the model and unzip it to the `./dependencies/` folder. 

Then, download the pretrained QA2D model from the Google Cloud [here](https://storage.cloud.google.com/few-shot-fact-verification/). You could download them to the `QA2D` folder using `gsutil`:

```shell
gsutil cp gs://few-shot-fact-verification/QA2D/* ./dependencies/QA2D/
```

Finally, run `generate_claim.sh` to generate claims from FEVER. Here is the example of generating NEI claims from the FEVER dev set. 

```shell
python Claim_Generation.py \
    --train_path ../data/fever_train.processed.json \
    --dev_path ../data/fever_train.processed.json \
    --entity_dict ../output/intermediate/entity_dict_train.json \
    --QA_path ../output/intermediate/precompute_QAs_train.json \
    --QA2D_model_path ../dependencies/QA2D_model \
    --sense_to_vec_path ../dependencies/s2v_old \
    --save_path ../output/NEI_claims.json \
    --range_start 0 \
    --range_end -1 \
    --claim_type NEI
```


## Zero-shot Fact Verification

Coming Soon...

## Reference
Please cite the paper in the following format if you use this dataset during your research.

```
@inproceedings{pan-etal-2021-Zero-shot-FV,
  title={Zero-shot Fact Verification by Claim Generation},
  author={Liangming Pan, Wenhu Chen, Wenhan Xiong, Min-Yen Kan, William Yang Wang},
  booktitle = {The Joint Conference of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (ACL-IJCNLP 2021)},
  address = {Online},
  month = {August},
  year = {2021}
}
```

## Q&A
If you encounter any problem, please either directly contact the first author or leave an issue in the github repo.


