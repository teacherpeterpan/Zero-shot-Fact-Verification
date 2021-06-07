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

Our processed dataset is publicly available in the Google Cloud Storage: [https://storage.cloud.google.com/few-shot-fact-verification/](https://storage.cloud.google.com/few-shot-fact-verification/)

You could download them to the `data` forder using `gsutil`:
```shell
gsutil cp gs://few-shot-fact-verification/* ./data/
```

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


