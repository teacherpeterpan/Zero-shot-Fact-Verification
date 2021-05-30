# Zero-shot-Fact-Verification-by-Claim-Generation
This repository contains code and models for the paper: [Zero-shot Fact Verification by Claim Generation (ACL-IJCNLP 2021)](). 

- We propose MQA-QG, an **unsupervised question answering** framework that can generate human-like multi-hop training pairs from both homogeneous and heterogeneous data sources. 

- We find that we can train a competent multi-hop QA model with only generated data. The F1 gap between the unsupervised and fully-supervised models is less than 20 in both the [HotpotQA](https://hotpotqa.github.io/) and the [HybridQA](https://hybridqa.github.io/) dataset.

- Pretraining a multi-hop QA model with our generated data would greatly reduce the demand for human-annotated training data for multi-hop QA. 

<p align="center">
<img src=Resource/operators.png width=700/>
</p>

## Requirements

- Python 3.7.3
- torch 1.7.1
- tqdm 4.49.0
- transformers 4.3.3
- stanza 1.1.1
- nltk 3.5
- dateparser 1.0.0
- scikit-learn 0.23.2
- fuzzywuzzy 0.18.0
