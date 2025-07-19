# 🧠 K-Nearest Neighbors (KNN) Project

This project demonstrates how the **K-Nearest Neighbors (KNN)** algorithm works for both **classification** and **regression** using Scikit-learn and Python.

---

## 📘 What is K-Nearest Neighbors (KNN)?

**KNN** is a simple and powerful **supervised machine learning algorithm** used for:
- **Classification**: Predicting a category
- **Regression**: Predicting a continuous value

KNN makes predictions by **comparing a new data point with its 'k' nearest data points** in the training set.

---

## 🧮 How KNN Works (Step-by-Step)

### 1. Choose the value of `k` (number of neighbors)

You decide how many nearby points to consider. A common choice is `k = 3` or `k = 5`.

### 2. Measure distance between the new point and all training points

The most common method is **Euclidean Distance**:

\[
d(p, q) = \sqrt{(p_1 - q_1)^2 + (p_2 - q_2)^2 + \cdots + (p_n - q_n)^2}
\]

Where:
- \( p = \) new data point
- \( q = \) a point in the training set
- \( n = \) number of features

### 3. Select the `k` closest points (lowest distance values)

### 4. Make prediction:
- **Classification**: Use majority vote (most common class)
- **Regression**: Take the average of the target values

---

## 📁 Project Structure

KNN/
├── KNN-classifier.ipynb   # KNN for classification
├── KNN-regressor.ipynb    # KNN for regression
├── README.md              # This file
└── images/                # Optional: for visual diagrams

---

## 📌 KNN Classifier Overview

**Notebook:** `KNN-classifier.ipynb`

- Dataset used: (Mention your dataset here)
- Steps:
  - Data cleaning
  - Feature scaling (important for KNN!)
  - Splitting into train/test
  - Choosing optimal `k` using **GridSearchCV**
  - Evaluation: Accuracy, Confusion Matrix, Classification Report

---

## 📌 KNN Regressor Overview

**Notebook:** `KNN-regressor.ipynb`

- Dataset used: (Mention your dataset here)
- Steps:
  - Feature normalization
  - Model fitting with `KNeighborsRegressor`
  - Evaluation using:
    - Mean Absolute Error (MAE)
    - Mean Squared Error (MSE)
    - R² Score

---

## 🔧 Hyperparameter Tuning

We used **GridSearchCV** to tune `k` from 1 to 10.  
Cross-validation ensures we avoid overfitting by validating on different subsets of data.

Example code:
```python
from sklearn.model_selection import GridSearchCV
params = {'n_neighbors': list(range(1, 11))}
grid = GridSearchCV(KNeighborsClassifier(), params, cv=5)
grid.fit(X_train, y_train)