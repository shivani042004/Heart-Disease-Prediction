# Heart Disease Prediction

## Overview

This project predicts whether a person has heart disease using their health data. I cleaned the dataset, explored it, and trained 5 different machine learning models to find the best one.

---

## Problem Statement

Heart disease is one of the leading causes of death worldwide. Predicting it early using basic patient data can help with faster diagnosis and better care.

---

## What I Did

* Loaded and checked the dataset for duplicates, nulls, and data types
* Found 0 duplicate rows and no missing values
* Fixed invalid 0 values in `Cholesterol` and `RestingBP` by replacing them with the column mean
* Explored the data with histograms, count plots, box plots, violin plots, and a correlation heatmap
* One-hot encoded categorical columns
* Scaled numerical features using StandardScaler
* Split data into train and test sets
* Trained and compared 5 models: Logistic Regression, KNN, Naive Bayes, Decision Tree, and SVM
* Saved the best model (KNN) along with the scaler and column list for future predictions

---

## Dataset

918 patient records with features like age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, max heart rate, exercise angina, oldpeak, and ST slope.

**Target column:** `HeartDisease` — 1 if the patient has heart disease, 0 if not

---

## Tools Used

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Joblib

---

## Key Findings

* 918 patients, no duplicates, no missing values
* About 55% of patients have heart disease — a fairly balanced dataset, good for training models fairly
* Cholesterol and RestingBP had invalid 0 values, fixed by replacing them with the column mean instead of dropping rows
* Heart disease is more common among male patients
* Patients with **asymptomatic chest pain (ASY)** have heart disease far more often than other chest pain types — a bit counterintuitive, since no obvious pain still means higher risk
* `Oldpeak` and `ExerciseAngina` are positively linked to heart disease; `MaxHR` is negatively linked (lower max heart rate → higher risk)
* `ST_Slope` is one of the strongest signals — a flat/downward slope means much higher heart disease risk than an upward slope

---

## Model Results

| Model               | Accuracy | F1 Score |
| ------------------- | -------- | -------- |
| Logistic Regression | 0.8750   | 0.8878   |
| **KNN**              | **0.8859**   | **0.8986**   |
| Naive Bayes         | 0.8696   | 0.8788   |
| Decision Tree       | 0.7391   | 0.7474   |
| SVM (RBF Kernel)    | 0.8641   | 0.8804   |

**Best model: KNN**, with the highest accuracy and F1 Score. It was saved as `KNN_heartdisease.pkl`, along with the fitted scaler and the column order, so it can be reused for new predictions without retraining.

Decision Tree performed the worst, likely because it overfit the training data instead of generalizing well.

---

## Repository Structure

```
Heart-Disease-Prediction/
├── heartdisease.ipynb          → Main notebook (this project)
├── heart.csv                   → Dataset
├── KNN_heartdisease.pkl        → Saved best model
├── scaler_heart.pkl            → Saved feature scaler
├── columns_heart.pkl           → Saved column order
├── README.md
└── Pluto heart_disease project/  → Additional analysis done during Pluto Academy internship
```

---

## What I Learned

* How to handle invalid zero-values properly (not just dropping them)
* How to explore health data using different chart types
* How to one-hot encode and scale features correctly
* How to compare multiple models fairly using accuracy and F1 Score
* How to save a trained model, scaler, and columns for future reuse

---

## Conclusion

This project covers a full machine learning workflow — cleaning messy medical data, exploring it visually, and testing 5 different models to find the most reliable one. KNN came out on top, correctly predicting heart disease in about 89% of cases, and was saved for future use.
