# Equitable AI for Dermatology Team Azealic Acid

## Overview
This project focuses on building a machine learning model that classifies various skin conditions across diverse skin tones, especially darker skin tones. Our goal in this project is to advance equity in dermatology by ensuring AI models are inclusive of populations historically underrepresented in medical AI datasets.

This challenge is from Break Through Tech and the Algorithmic Justice League to help address the issue of diagnosing people with darker skin tones by building an inclusive machine learning model for dermatology.

- [FitzPatrick 17k Dataset Paper](https://arxiv.org/abs/2104.09957)
- [Kaggle Competition Page](https://www.kaggle.com/competitions/bttai-ajl-2025/overview)

## Task: Skin Condition Classification

The goal is to develop an ML model that accurately classifies skin conditions using the provided dataset. The competition is evaluated based on the weighted average F1 score, which balances precision and recall across different class labels.
- Weighted F1 Score: Ensures performance is measured fairly across all classes, accounting for class imbalance.
    - F1 = 2 * (precision * recall) / (precision + recall)


## Dataset

The dataset used is a subset of the FitzPatrick17k dataset, which contains ~17,000 labeled images of different dermatological conditions. The competition dataset consists of 4,500 images representing 21 skin conditions across skin tones that are categorized by the FitzPatrick Skin Type Scale (FST).
- Source: DermaAmin & Atlas Dermatologico (dermatology image repositories)
- Skin tone labels: Self-described & externally labeled FST scores


## Getting Started

1) Download dataset 

```
kaggle competitions download -c bttai-ajl-2025
```
