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

## Data Preparation

The data used in our paper is constructed based on the original [FEVER dataset](https://fever.ai/resources.html). We use the gold evidence sentences in FEVER for the SUPPORTED and REFUTED claims. We collect evidence sentences for the NEI class using the retrival method proposed in [**the Papelo system from FEVER'2018**](https://github.com/cdmalon/fever2018-retrieval). The detailed data processing process is introduced [here](./data_processing.md). 

Our processed dataset is publicly available in the Google Cloud Storage: [https://storage.cloud.google.com/few-shot-fact-verification/](https://storage.cloud.google.com/few-shot-fact-verification/)

You could download them to the `data` folder using `gsutil`:
```shell
gsutil cp gs://few-shot-fact-verification/* ./data/
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

### Pre-Processing

First, we extract all Name Entities (NERs) in the evidence. 

```shell
mkdir -p ../output/intermediate/

python Extract_NERs.py \
    --train_path ../data/fever_train.processed.json \
    --dev_path ../data/fever_dev.processed.json \
    --save_path ../output/intermediate/
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


