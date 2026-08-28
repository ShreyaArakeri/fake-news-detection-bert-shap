# Fake News Detection: Combining Deep Learning with Linguistic Feature Explainability

**MSc Dissertation**

## Project Overview
This project develops and evaluates a fake news detection system 
using transformer-based deep learning models with SHAP explainability 
analysis across three benchmark datasets.

## Models Evaluated
- Logistic Regression (baseline)
- Support Vector Machine (baseline)
- DistilBERT
- BERT-base
- RoBERTa-base

## Datasets
- [Kaggle Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- [LIAR Political Claims Dataset](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip)
- [WELFake Dataset](https://zenodo.org/record/4561253)

## Key Results

| Model | Kaggle | LIAR | WELFake |
|---|---|---|---|
| Logistic Regression | 97.3% | 62.4% | 90.2% |
| SVM | 97.8% | 61.2% | 92.1% |
| DistilBERT | 99.3% | 63.7% | 99.1% |
| BERT-base | 99.7% | 65.7% | 98.7% |
| RoBERTa-base | 99.8% | 43.6%* | 97.6% |

*RoBERTa exhibited class collapse on LIAR

## Notebooks

| Notebook | Environment | Description |
|---|---|---|
| `baseline_models.ipynb` | Jupyter (laptop CPU) | Logistic Regression and SVM classifiers |
| `three_dataset_combined_v2.ipynb` | Google Colab (T4 GPU) | Transformer fine-tuning pipeline |
| `shap_explainability_v3.ipynb` | Google Colab (T4 GPU) | SHAP explainability analysis |

> **Note:** Transformer notebooks require Google Colab with GPU 
> enabled (Runtime → Change runtime type → T4 GPU)

## Requirements
pip install transformers torch scikit-learn shap
textblob wordcloud pandas numpy matplotlib


## Key Findings
- BERT-base achieved best overall performance across all three datasets
- SHAP analysis revealed fake news uses informal emotional language
- Real news consistently uses formal institutional vocabulary
- Source-identity leakage identified and mitigated in Kaggle dataset
