# Predictive Maintenance Analysis (UCI AI4I 2020)

## Project Overview
This project applies machine learning techniques to the **UCI AI4I 2020 Predictive Maintenance Dataset**, a real-world dataset reflecting the physical properties of industrial milling machines.

The goal is to build a robust classification model capable of predicting machine failures based on sensor readings (Air Temperature, Process Temperature, Rotational Speed, Torque, and Tool Wear), enabling proactive maintenance strategies.

## Workflow & Key Techniques

### 1. Data Pipeline
*   **Data Source**: Integrated the UCI AI4I 2020 dataset directly via URL.
*   **Data Cleaning**: Removed non-predictive identifiers (UID, Product ID) and prevented **data leakage** by excluding specific failure type columns (e.g., TWF, HDF) to focus solely on sensor-based prediction.
*   **Feature Engineering**: Applied One-Hot Encoding to categorical quality variants (Type L/M/H).

### 2. Handling Imbalance
*   **Stratified Splitting**: Used `stratify=y` during train/test split to maintain the failure ratio.
*   **SMOTE (Synthetic Minority Over-sampling Technique)**: Applied SMOTE **only to the training set** to synthesize minority class samples, ensuring the model learns failure patterns without biasing the validation results.

### 3. Modeling & Evaluation
*   **Algorithm**: Random Forest Classifier (optimized with `n_jobs=-1`).
*   **Metrics**: Evaluated using Precision, Recall, F1-Score, and **ROC-AUC** to handle the trade-off between false alarms and missed failures.
*   **Visualization**: Generated Confusion Matrices and Feature Importance plots to interpret model behavior.

## Results
The model demonstrates realistic performance on unseen test data. Feature Importance analysis indicates that **Torque** and **Rotational Speed** are the primary predictors of machine failure, consistent with mechanical engineering principles.

## How to View
Click on the file `Predictive_Maintenance_Analysis.ipynb` to view the complete code, data processing steps, and visualization outputs.
