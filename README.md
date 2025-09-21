# Cardiovascular Risk Prediction

Myocardial Infarction Risk Prediction using KNN
This project applies machine learning to predict the risk of myocardial infarction (heart attack). The focus is on handling severe class imbalance, performing medical feature engineering, and ensuring predictions are both accurate and clinically interpretable.

•	Project Highlights
Built a K-Nearest Neighbors (KNN) classifier on 13,768 NHANES patient records.
Engineered features such as cholesterol ratios and age–risk factor interactions to capture medical relevance.
Addressed a 24:1 class imbalance using resampling strategies.
Achieved 78% balanced accuracy and F1-score of 0.78 for both classes.
Evaluated performance using accuracy, F1-score, balanced accuracy, and KL divergence.
Final output: patient ID with predicted probability of MI risk.

• Dataset
The dataset comes from NHANES, containing demographic, lifestyle, and clinical variables.
Train set: 13,768 samples with labels (MI / Non-MI).
Test set: Held-out dataset with patient IDs for probability prediction.

• Tech Stack
Python: pandas, numpy, scikit-learn, matplotlib, imblearn
Model: KNN Classifier
Methods: Feature engineering, class imbalance handling, probability calibration

• Evaluation Metrics
To ensure fair and balanced evaluation across highly imbalanced classes:
Balanced Accuracy
F1-score
Kullback–Leibler (KL) Divergence
Confusion Matrix Analysis

• Workflow
Data Preprocessing
Missing value imputation (KNN imputer)
Scaling and normalization
Feature engineering (cholesterol ratios, BMI interactions, etc.)
Class Imbalance Handling
Undersampling the majority class
Ensuring balanced distribution during training
Model Training
K-Nearest Neighbors with tuned hyperparameters
Validation set for model selection
Evaluation
Accuracy, F1-score, KL divergence
Class distribution of predictions
Patient ID + Predicted Probability of MI risk

• Results
Balanced Accuracy: 78%
F1-score (MI): 0.77
F-1 score (Non-MI) : 0.78
Predictions remain balanced across both classes with calibrated probabilities.
