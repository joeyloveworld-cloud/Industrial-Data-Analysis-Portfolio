# Predictive Maintenance Analysis for Industrial Milling Machines

## Project Overview
This project demonstrates the application of machine learning techniques to solve a classic problem in Industrial Engineering: **Predictive Maintenance**. 

Using simulated sensor data from a milling machine (RPM, Torque, Tool Wear, Temperature), I developed a classification model to predict potential machine failures before they occur.

## Key Techniques Used
*   **Data Simulation**: Generated realistic sensor data based on physical constraints.
*   **Data Preprocessing**: Handling imbalanced datasets using **SMOTE** (Synthetic Minority Over-sampling Technique).
*   **Modeling**: Implemented a **Random Forest Classifier** for robust prediction.
*   **Evaluation**: Analyzed model performance using Confusion Matrix and Recall scores.

## Results
The model successfully identified key indicators of failure. The Feature Importance analysis confirmed that **Tool Wear (min)** and **Torque (Nm)** are the most critical factors, aligning with physical mechanical principles.

## How to View
Click on the file `Predictive_Maintenance_Analysis.ipynb` above to view the full code, analysis process, and visualization charts.
