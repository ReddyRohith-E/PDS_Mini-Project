# Breast Cancer Prediction: A Machine Learning Approach

## Team Members
- 22691A05I2
- 22691A05I4
- 22691A05I5
- 22691A05I6
- 22691A05I7

**Team Leader:** 22691A05I1 

---

## Problem Statement
Breast cancer is a leading cause of mortality among women worldwide, with early detection playing a critical role in improving survival rates. Traditional diagnostic methods rely heavily on manual interpretation, leading to potential delays and inaccuracies. Machine learning provides a promising avenue for enhancing diagnostic precision and efficiency.

The primary goal of this project is to build and evaluate machine learning models that can predict breast cancer based on clinical data. Using the Wisconsin Breast Cancer Dataset (WBCD), this project employs algorithms like Naive Bayes and Support Vector Machine (SVM) to build accurate predictive models. The project also explores feature importance through ablation studies and optimizes models using hyperparameter tuning, with performance measured through metrics like accuracy, precision, recall, and F1-score.

---

## Objective
The overarching objective is to harness machine learning for accurate breast cancer prediction. Specific aims include:
- Developing predictive models based on WBCD to distinguish between benign and malignant tumors.
- Exploring feature importance through systematic ablation to identify impactful clinical factors.
- Performing hyperparameter tuning to improve model accuracy and robustness.
- Validating the proposed approach through standard evaluation metrics.

This project demonstrates the potential of machine learning in improving diagnostic workflows, with implications for personalized treatment and early detection of breast cancer.

---

## Proposed Method

### Workflow
1. **Dataset Collection:** Wisconsin Breast Cancer Dataset with 569 samples and 30 features.
2. **Preprocessing:** Handle missing values, normalize data, and standardize features.
3. **Visualization:** Perform exploratory analysis to identify trends and distributions.
4. **Feature Engineering:** Enhance model performance and interpretability.
5. **Modeling:** Train and evaluate Naive Bayes and SVM algorithms.
6. **Performance Metrics:** Assess models using accuracy, precision, recall, and F1-score.

### Dataset Collection
The WBCD is a reliable dataset widely used in breast cancer research, containing tumor characteristics like radius, texture, and smoothness. The dataset was accessed via the Scikit-learn library.  
**Dataset Link:** [Scikit-learn WBCD Documentation](https://scikit-learn.org/stable/datasets/index.html#wisconsin-breast-cancer-dataset)

### Data Preprocessing
- **Handling Missing Values:** Imputation techniques were applied for missing data.
- **Normalization:** Data scaled to a consistent range using Min-Max Scaling.
- **Standardization:** Features follow a standard normal distribution for better algorithm compatibility.

### Data Visualization
- Histogram of Tumor Radius Distribution.
- Boxplot of Smoothness by Diagnosis (Benign/Malignant).
- Correlation Heatmap of Features.

---

## Machine Learning Algorithms

### Naive Bayes
**Description:** A probabilistic classifier based on Bayes' theorem. Assumes features are conditionally independent.  

**Advantages:**
- Computationally efficient.
- Performs well on small datasets.

**Limitations:**
- Sensitive to the independence assumption.
- Struggles with numerical features without proper preprocessing.

### Support Vector Machine (SVM)
**Description:** A powerful supervised learning algorithm designed for classification and regression tasks. Finds the optimal hyperplane for class separation.  

**Advantages:**
- High accuracy in high-dimensional spaces.
- Robust against overfitting when parameters are correctly tuned.

**Limitations:**
- Computationally intensive for large datasets.
- Requires careful hyperparameter tuning.

---

## Results and Discussion

### Algorithm 1: Naive Bayes
**Performance Metrics:**
- Accuracy: 0.9561 (95.61%)
- Precision: 0.9623 (96.23%)
- Recall: 0.9623 (96.23%)
- F1-Score: 0.9623 (96.23%)
- ROC AUC: 0.9965

**Confusion Matrix:**
|               | Predicted Benign | Predicted Malignant |
|---------------|------------------|---------------------|
| **Actual Benign** | 87               | 3                   |
| **Actual Malignant** | 2               | 111                 |

**Observations:**
- Naive Bayes achieved high performance metrics.
- Minimal false positives (3) and false negatives (2).

### Algorithm 2: SVM
**Performance Metrics:**
- Accuracy: 0.9825 (98.25%)
- Precision: 0.9737 (97.37%)
- Recall: 1.0000 (100%)
- F1-Score: 0.9867 (98.67%)

**Confusion Matrix:**
|               | Predicted Benign | Predicted Malignant |
|---------------|------------------|---------------------|
| **Actual Benign** | 90               | 0                   |
| **Actual Malignant** | 0               | 113                 |

**Observations:**
- SVM outperformed Naive Bayes in all metrics.
- Perfect recall with no false negatives.

---

## Comparison and Justification
**Performance Comparison Table:**
| Metric       | Naive Bayes | SVM        |
|--------------|-------------|------------|
| Accuracy     | 95.61%      | 98.25%     |
| Precision    | 96.23%      | 97.37%     |
| Recall       | 96.23%      | 100%       |
| F1-Score     | 96.23%      | 98.67%     |

SVM demonstrated superior performance due to its ability to handle non-linear relationships and provide optimal separation between classes. While Naive Bayes is computationally efficient, its independence assumption likely limited its performance.

---

## Conclusion
This project demonstrated that machine learning can effectively improve breast cancer diagnosis. SVM emerged as the superior algorithm, achieving higher performance metrics and better handling of non-linear relationships. Future directions include exploring deep learning models, integrating additional datasets, and enhancing model interpretability using techniques like SHAP or LIME.

---

## Contributors
| Name | Roll Number | Responsibility |
|------|-------------|----------------|
| E. Reddy Rohith | 22691A05I1 | Team Leader, Model Implementation |
| V. Reddy Salma | 22691A05I2 | Data Preprocessing, Feature Engineering |
| S. Rehana Banu | 22691A05I4 | Data Visualization, Documentation |
| P. Ruthik | 22691A05I5 | Algorithm Selection, Analysis |
| S. Rukkiya Nashad | 22691A05I6 | Performance Evaluation, Reporting |
| K. Ruksana | 22691A05I7 | Code Integration, Debugging |

---

## References
- Megha Rathi, Arun Kumar Singh. *"Breast Cancer Prediction using Naïve Bayes Classifier."*
- Wisconsin Breast Cancer Dataset: [Scikit-learn Documentation](https://scikit-learn.org/stable/datasets/index.html#wisconsin-breast-cancer-dataset).
- *"Support Vector Machines in Medicine: Applications and Case Studies."*
- *"Data Preprocessing Techniques for Medical Datasets."*
- *"A Guide to Hyperparameter Tuning for Machine Learning Models."*

---

**GitHub Repository:** [Breast Cancer Prediction Code](https://github.com/ReddyRohith-E/PDS_Mini-Project.git)
