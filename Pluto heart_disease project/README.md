# Heart Disease Prediction using Machine Learning

## Overview

This project predicts whether a person has heart disease based on their health data. I cleaned the dataset, explored it, and trained multiple machine learning models to find the best one.

---

## Problem Statement

Heart disease is one of the biggest causes of death worldwide. Early prediction can help doctors and patients act sooner. This project uses patient health data to predict heart disease risk using machine learning.

---

## What I Did

* Loaded and checked the dataset for duplicates and missing values
* Removed 723 duplicate rows
* Explored the data with histograms, count plots, box plots, violin plots, and a heatmap
* One-hot encoded categorical columns (chest pain type, ECG, slope, thal)
* Scaled numerical features using StandardScaler
* Split data into train and test sets
* Trained and compared 3 models: Logistic Regression, Random Forest, and KNN
* Picked the best model using F1 Score and checked it with a confusion matrix

---

## Dataset

Patient-level health data with features like age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, ECG results, max heart rate, exercise-induced angina, and more.

**Target column:** `target` — whether the patient has heart disease(1) or not(0)

---

## Tools Used

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

---

## Key EDA Findings

* Males are affected by heart disease more than females.
* Most heart disease patients fall in the 40–60 age group, while non-disease patients skew slightly older.
* No missing values in the dataset.
* Found 723 duplicate rows out of 1,748 (over 40% of the raw data), removed before analysis.
* After cleaning: 1,025 unique patients, ages 29–77 (average ~54), cholesterol ranging from 126–564 mg/dl.
* Chest pain type, max heart rate, and slope show a positive relationship with heart disease risk.
* Exercise-induced angina, oldpeak, number of major vessels (ca), and thal show a negative relationship with heart disease risk.

---

## Model Results

| Model               | Accuracy | Precision | Recall | F1 Score |
| ------------------- | -------- | --------- | ------ | -------- |
| **Logistic Regression** | 0.8197   | 0.8235    | 0.8485 | **0.8358** |
| Random Forest        | 0.8033   | 0.8387    | 0.7879 | 0.8125   |
| KNN                  | 0.7869   | 0.8125    | 0.7879 | 0.8000   |

**Best model: Logistic Regression**

---

## Best Model Analysis

Logistic Regression came out as the best model, with the highest F1 Score (0.8358) and best recall (0.8485), meaning it caught the most real heart disease cases correctly, the most important thing in a medical prediction task, since missing a sick patient is worse than a false alarm. Random Forest and KNN scored slightly lower, especially on recall, so they missed more true cases. This happened because the dataset is small, clean, and mostly linear, so a simpler model like Logistic Regression generalized better instead of overfitting, as the more complex models can. Overall, the project shows that basic patient health data (age, cholesterol, blood pressure, etc.) can predict heart disease risk, with Logistic Regression giving the most reliable ~82% accurate predictions.

---

## What I Learned

* How to clean and de-duplicate real-world health data
* How to one-hot encode categorical variables correctly
* How to scale features before training models
* How to train and compare multiple classification models
* How to pick the best model using proper metrics (not just accuracy)
* How to read a confusion matrix

---

## Conclusion

This project shows a complete machine learning workflow, from cleaning data to comparing models to picking the best one using the right metric. Logistic Regression turned out to be the most reliable model for this dataset, showing that simpler models can beat complex ones when the data is small and clean.
