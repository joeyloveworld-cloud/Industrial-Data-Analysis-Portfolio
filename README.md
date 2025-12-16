# Steel Plates Faults Classification 

## Project Overview
This project implements a robust machine learning pipeline for the **UCI Steel Plates Faults Dataset**. The objective is to correctly classify seven distinct types of surface defects in steel plates (e.g., Pastry, Z_Scratch, K_Scatch) using geometric and texture features.

The solution focuses on handling high-dimensional sensor data, addressing class imbalance, and maximizing classification accuracy through ensemble learning methods.

## Workflow & Key Techniques

### 1. Data Pipeline
*   **Data Source**: Automated retrieval from the UCI Machine Learning Repository.
*   **Data Cleaning**: 
    *   Filtered out "noise" rows where the sum of target labels was not exactly 1 (removing ambiguous or non-fault cases).
    *   Converted the original One-Hot Encoded targets into a single categorical target variable for efficient modeling.
*   **Feature Engineering**: 
    *   **Collinearity Removal**: Automatically detected and removed features with a correlation coefficient > 0.95 to reduce redundancy and improve model stability.
    *   **Standardization**: Applied `StandardScaler` to normalize feature distributions, ensuring compatibility with linear components of the ensemble.

### 2. Modeling Strategy
*   **Ensemble Architecture**: Implemented a `VotingClassifier` (Soft Voting) combining three distinct algorithms to leverage their complementary strengths:
    *   **LightGBM**: For high efficiency and handling non-linear patterns.
    *   **Random Forest**: For robustness against overfitting and variance reduction.
    *   **Logistic Regression**: To capture linear decision boundaries.
*   **Imbalance Handling**: Utilized `class_weight='balanced'` across all models to penalize misclassification of minority fault types.
*   **Validation**: Employed Stratified K-Fold Cross-Validation (5 folds) to ensure reliable performance estimates.

### 3. Evaluation & Visualization
*   **Metrics**: Evaluated using Macro F1-Score (crucial for multi-class imbalance) and ROC-AUC.
*   **Visualization**: 
    *   **Confusion Matrix**: Generated heatmaps to identify specific misclassification patterns between similar fault types.
    *   **ROC Curves**: Plotted One-vs-Rest ROC curves to analyze the model's discrimination capability for each specific fault class.

## Results
The optimized ensemble model achieves strong performance on the unseen test set:

*   **Macro F1-Score**: ~0.80+
*   **Balanced Accuracy**: ~0.82+

> **Note**: The confusion matrix and ROC curves are automatically generated in the `outputs/` directory upon running the script.

## How to Run

1.  **Install Dependencies**:
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn lightgbm joblib
    ```

2.  **Execute the Pipeline**:
    ```bash
    python main.py
    ```

3.  **View Outputs**:
    Check the `outputs/` folder for:
    *   `confusion_matrix.png`
    *   `roc_curves.png`
    *   `steel_fault_model.pkl` (Saved Model)
