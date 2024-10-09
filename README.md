# Unified-Classifier-Stacking-A-Comprehensive-Ensemble-Approach-to-Cyberbullying-Detection
This project focuses on detecting cyberbullying in social media posts using a stacked ensemble learning approach. The model has been designed to classify text data efficiently while addressing challenges posed by imbalanced datasets and nuanced language patterns associated with cyberbullying.

## Project Overview

The model leverages a combination of machine learning techniques and data preprocessing steps to achieve high accuracy and robustness in detecting cyberbullying. The ensemble approach improves generalization by combining multiple classifiers into a stronger predictive model.

### Key Features:
- **Ensemble Learning**: The model integrates **Decision Tree (DT)**, **Support Vector Machine (SVM)**, and **K-Nearest Neighbors (KNN)** as base learners, with **Logistic Regression (LR)** serving as the meta-learner.
- **Data Preprocessing**: 
  - Text is processed by applying techniques such as **lowercasing**, **stopword removal**, and **stemming**.
  - **TF-IDF vectorization** is used for feature extraction.
  - **Random Over Sampling (ROS)** addresses the issue of class imbalance in the training data.
  - **K-Fold cross-validation** is utilized for model training and evaluation to ensure robustness and reduce overfitting.

## Model Performance

The model was trained on a Twitter post dataset and achieved the following results:
- **Accuracy**: 91.43%
- **Precision**: 91.99%
- **Recall**: 90.46%
- **F1-Score**: 91.22%
- **AUC (Area Under Curve)**: 96.77%

These metrics validate the model's effectiveness in detecting cyberbullying instances from social media text data, even in the presence of imbalanced data distributions.

## Techniques & Tools Used

- **Machine Learning Algorithms**: 
  - Decision Tree (DT)
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Logistic Regression (LR)
  
- **Data Preprocessing**: 
  - TF-IDF Vectorization
  - Stopword Removal, Stemming
  - Random Over Sampling (ROS)
  
- **Model Evaluation**: 
  - K-Fold Cross-Validation
  - Precision, Recall, F1-Score, AUC
  
## Improvements and Generalizability

By leveraging a diverse set of algorithms in an ensemble learning framework, the model improves both prediction accuracy and generalizability. This approach also addresses challenges posed by the imbalanced nature of the cyberbullying dataset, making it more robust in real-world applications.

## Conclusion

This cyberbullying detection model showcases the power of ensemble learning and effective data preprocessing. The high performance metrics (accuracy, precision, recall, F1-score, and AUC) demonstrate its ability to classify cyberbullying effectively, contributing to the creation of safer digital environments.

