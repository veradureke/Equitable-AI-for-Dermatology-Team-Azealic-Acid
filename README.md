# Equitable AI for Dermatology Team Azealic Acid
> Team Members: Janelle Chan, Vera Dureke, Ena Macahiya, Natasha Prabhoo, Shreya Vallala

## Overview
Our team participated in the Equitable AI for Dermatology Kaggle competition with the goal of training a model that could classify different skin conditions across diverse skin tones, using a provided subset of the FitzPatrick17k dataset. This smaller dataset contains approximately 4,500 images representing 21 skin conditions from the total 100+ in the full dataset. The motivation for this was the fact that dermatology AI tools underperform for people with darker skin tones due to a lack of training data. The unfortunate impact includes diagnostic errors, delayed treatments, and health disparities for underserved communities.

- [FitzPatrick 17k Dataset Paper](https://arxiv.org/abs/2104.09957)
- [Kaggle Competition Page](https://www.kaggle.com/competitions/bttai-ajl-2025/overview)

## Data Exploration
The dataset contains information related to dermatological conditions. It includes both numerical values (e.g., Fitzpatrick scale, DDI scale) and categorical features (e.g., skin condition labels, classifications into "benign", "malignant", or "non-neoplastic").

In this EDA, we focus on visualizing and analyzing the distributions of important features, identifying missing values, and checking the relationships between categorical variables and encoded numerical values. We were also provided information about our dataset, which can be found on the [Kaggle competition page](https://www.kaggle.com/competitions/bttai-ajl-2025/data). 

**Null Values**: We began by identifying missing values. Upon inspection, the qc column has many missing values across both training and testing datasets.
nan_count = np.sum(train_df.isnull(), axis = 0)

**Fitzpatrick Scale and Centaur Distribution**: We visualized the distribution of the fitzpatrick_scale and fitzpatrick_centaur columns using histograms. This reveals the frequency of various Fitzpatrick skin types across the dataset. One of the main competition goals was to improve model accuracy by looking into the distribution of representation within the dataset. Having lesser images of certain classes of skin tones will ultimately affect those populations and the predictions made upon them. 
```
frequency of ints in fitzpatrick_scale:
fitzpatrick_scale
 2    964
 3    562
 1    528
 4    393
 5    216
-1    108
 6     89
Name: count, dtype: int64

frequency of each integer in fitzpatrick_centaur:
fitzpatrick_centaur
 1    1000
 2     730
 3     489
 4     253
-1     161
 5     147
 6      80
Name: count, dtype: int64
```
<a href="url"><img src="https://github.com/user-attachments/assets/bbd89e68-02ad-4555-9f11-d984cf5c5be5" align="middle" height=40% width=40% ></a>
<a href="url"><img src="https://github.com/user-attachments/assets/0614b8ab-380a-46b6-9ce1-3900c4d9d42c" align="middle" height=40% width=40% ></a>

From these histograms, we can see that lighter skin tones are significantly more present in the dataset than others.

**Label Distribution:** We also explored the distribution of skin condition labels using bar plots. This helps us understand the distribution of skin conditions across the dataset:
The bar charts highlight the dominant skin condition categories and reveal any imbalances in the dataset, which may be important for modeling.

**Correlation Analysis**
We encoded categorical variables (like label, nine_partition_label, and three_partition_label) into numerical values using LabelEncoder and calculate the correlation matrix to understand the relationships between them.
The correlation matrix reveals that:
- There is a mild positive correlation between label_encoded and nine_partition_encoded (0.20).
- There is a moderate negative correlation between label_encoded and three_partition_encoded (-0.44).
- The correlation between nine_partition_encoded and three_partition_encoded is weak (-0.11).
These correlations provide insight into the relationships between different skin condition classifications.

**Conclusion**
In this EDA, we performed several key steps:
- Analyzed the distribution of key features such as fitzpatrick_scale and fitzpatrick_centaur.
- Explored the frequency of skin condition labels and partitions.
- Calculated and visualized correlations between different encoded categorical variables.
The insights from this analysis provide a foundation for building machine learning models to classify skin conditions based on these features. Future steps could involve feature engineering, model training, and evaluating predictive accuracy.

### Getting Started

1) Download dataset 

```
kaggle competitions download -c bttai-ajl-2025
```
