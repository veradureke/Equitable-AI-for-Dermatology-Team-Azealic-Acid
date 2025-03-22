# Equitable AI for Dermatology Team Azealic Acid
> Team Members: Janelle Chan, Vera Dureke, Ena Macahiya, Natasha Prabhoo, Shreya Vallala

## Overview
Our team participated in the Equitable AI for Dermatology Kaggle competition with the goal of training a model that could classify different skin conditions across diverse skin tones, using a provided subset of the FitzPatrick17k dataset. This smaller dataset contains approximately 4,500 images representing 21 skin conditions from the total 100+ in the full dataset. The motivation for this was the fact that dermatology AI tools underperform for people with darker skin tones due to a lack of training data. The unfortunate impact includes diagnostic errors, delayed treatments, and health disparities for underserved communities.

- [FitzPatrick 17k Dataset Paper](https://arxiv.org/abs/2104.09957)
- [Kaggle Competition Page](https://www.kaggle.com/competitions/bttai-ajl-2025/overview)

### Getting Started

1) Download dataset 

```
kaggle competitions download -c bttai-ajl-2025
```

## Data Exploration
The dataset contains information related to dermatological conditions. It includes both numerical values (e.g., Fitzpatrick scale, DDI scale) and categorical features (e.g., skin condition labels, classifications into "benign", "malignant", or "non-neoplastic").

In this EDA, we focus on visualizing and analyzing the distributions of important features, identifying missing values, and checking the relationships between categorical variables and encoded numerical values. We were also provided information about our dataset, which can be found on the [Kaggle competition page](https://www.kaggle.com/competitions/bttai-ajl-2025/data). 

### Null Values 
We began by identifying missing values. Upon inspection, the qc column has many missing values across both training and testing datasets.
```nan_count = np.sum(train_df.isnull(), axis = 0)```

### Fitzpatrick Scale and Centaur Distribution 
We visualized the distribution of the fitzpatrick_scale and fitzpatrick_centaur columns using histograms. This reveals the frequency of various Fitzpatrick skin types across the dataset. One of the main competition goals was to improve model accuracy by looking into the distribution of representation within the dataset. Having lesser images of certain classes of skin tones will ultimately affect those populations and the predictions made upon them. 
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
<br /> 
<a href="url"><img src="https://github.com/user-attachments/assets/bbd89e68-02ad-4555-9f11-d984cf5c5be5" align="middle" height=40% width=40% ></a>
<a href="url"><img src="https://github.com/user-attachments/assets/0614b8ab-380a-46b6-9ce1-3900c4d9d42c" align="middle" height=40% width=40% ></a>
<br /> 

From these histograms, we can see that lighter skin tones are significantly more present in the dataset than others.

### Label Distribution
We also explored the distribution of skin condition labels using bar plots. This helps us understand the distribution of skin conditions across the dataset:
The bar charts highlight the dominant skin condition categories and reveal any imbalances in the dataset, which may be important for modeling.
<br /> 
<a href="url"><img src="https://github.com/user-attachments/assets/e89db65a-3550-4f47-83b3-c7b6a13f1e55" align="middle" height=40% width=40% ></a>
<a href="url"><img src="https://github.com/user-attachments/assets/6b65a2f1-4d7a-4204-8fa3-e3590bcdf5c5" align="middle" height=40% width=40% ></a>
<br /> 
### Correlation Analysis
We encoded categorical variables (like label, nine_partition_label, and three_partition_label) into numerical values using LabelEncoder and calculate the correlation matrix to understand the relationships between them.
The correlation matrix reveals that:
- There is a mild positive correlation between label_encoded and nine_partition_encoded (0.20).
- There is a moderate negative correlation between label_encoded and three_partition_encoded (-0.44).
- The correlation between nine_partition_encoded and three_partition_encoded is weak (-0.11).
These correlations provide insight into the relationships between different skin condition classifications.
<br /> 
<a href="url"><img src="https://github.com/user-attachments/assets/22c06a91-40e2-4a7e-8d97-ed500062be28" align="middle" height=50% width=50% ></a>
<br />

### EDA Conclusion 
In this EDA, we performed several key steps:
- Analyzed the distribution of key features such as fitzpatrick_scale and fitzpatrick_centaur.
- Explored the frequency of skin condition labels and partitions.
- Calculated and visualized correlations between different encoded categorical variables.
The insights from this analysis provide a foundation for building machine learning models to classify skin conditions based on these features. Future steps could involve feature engineering, model training, and evaluating predictive accuracy.

## Model Development
The model we decided to start with for this task was ResNet50. This was recommended to us by our TA as a pretty standard model for image classification problems. This model is a 50-layer residual network that uses residual connections (skip connections) to help in training deeper networks by mitigating the vanishing gradient problem. It allows gradients to flow more easily during backpropagation, leading to overall better convergence. The model is pre-trained on ImageNet, making it efficient for transfer learning. 

Later in the competition, we decided to also try out the [Google Derm Foundation model on HuggingFace](https://huggingface.co/google/derm-foundation) in hopes of achieving a higher accuracy. This model is a pre-trained BiT-101x3 CNN designed to accelerate AI development for dermatology image analysis. It generates 6144-dimensional embeddings from skin images, capturing dense features crucial for classification tasks. This model was pre-trained using contrastive learning on large-scale image-text pairs and is tailored for dermatology using clinical datasets. 

## Impact Narrative
Dermatology AI tools frequently underperform for people with darker skin tones due to a lack of diverse training data, as most datasets are primarily composed of images from lighter-skin tones as we saw during our EDA. As a result, these models have lower accuracy in detecting skin conditions in people with darker skin, leading to patient misdiagnosis, delayed treatments, and worse health outcomes. When AI systems perpetuate these inequities, they continue to promote historical biases and further hurt marginalized communities.

Our project seeks to address these disparities by training a model that classifies skin conditions equitably across diverse skin tones. To do this, we did the following: 
-  By using image augmentation, we can even out the representation of skin tones through rotatings, transformations, etc.
- We use visualizations and interpretability techniques to understand how our model makes predictions, uncovering / correcting any biases in decision-making.

This project is a part of a larger movement towards ethical and responsible AI in healthcare. By highlighting the real-world consequences of biased AI and working towards solutions, we hope to contribute to systemic change in dermatology and beyond.

## Next Steps and Further Improvements
The next steps for this project would include further technical development and deeper fairness analysis to improve the model's performance and accuracy. We could experiment with more advanced architectures and, on the data side, create a more indepth and balanced dataset for the model to diagnose accurately across a variety of skin tones. When we have a fully functional product that has gone through real-world testing we can integrate the product with dermatologists in local clinics to validate patient diagnoses in real-world appointments. 
