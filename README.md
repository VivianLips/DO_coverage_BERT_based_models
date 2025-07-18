# Automatic coverage evaluation of the Disease Ontology (D0) using transformer models

## Overview
This repository contains the code used for the thesis: "Ontology Coverage Evaluation using BERT-based Transformer Models" (Vrije Universiteit Amsterdam, 2025).

In this research, we propose an automated framework for evaluating ontology coverage for the Disease Ontology by automatically extracting disease entities from a corpus of biomedical literatur using different transformer models (BioBERT, PubMedBET, ClinicalBERT) fine-tuned on various levels (NCBI-disease corpus, BC5CDR corpus, a combined dataset). The extracted entities are aligned to the Disease Ontology to compute evaluation metrics to assess coverage. Additionally a placement framework was constructed to suggest placement of novel identified terms.

## Features
- Corpus collection and preprocessing using PMC IDs
- Model fine-tuning and evaluation
- Disease entity extraction from domain-specific corpus using fine-tuned models
- Extracted entity alignment to the Disease Ontology
- Evaluation of coverage metrics across models, fine-tuning, and corpus size
- Entity placement framework using Cosine similarity and Natural Language Inference

## Requirement
Main libraries used:
- Transformers (Huggingface)
- spaCy
- PyTorch
- NLTK
- SentenceTransformers
- Pandas
- skicit-learn
- request
- BeautifulSoup

## Datasets & Ontology
- NCBI: https://www.ncbi.nlm.nih.gov/research/bionlp/Data/disease/
- BC5CDR: https://huggingface.co/datasets/bigbio/bc5cdr
- Disease Ontology: https://www.disease-ontology.org/do/

## Citation
If you use this code in your work, please cite:
@article{lips2025ontologycoverage,
  title={Ontology Coverage Evaluation using BERT-based Transformer Models},
  author={Vivian  Lips},
  year={2025}
}
