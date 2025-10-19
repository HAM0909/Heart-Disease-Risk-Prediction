# Heart Disease Risk Prediction using PySpark ML

## Overview

This project implements a machine learning pipeline for predicting heart disease risk using PySpark MLlib. The notebook demonstrates comprehensive data preprocessing, feature engineering, and model comparison using three different classification algorithms.

## Dataset

The dataset contains medical information for 297 patients after cleaning (6 rows were dropped due to missing values), with the following features:

### Original Features
- **ID**: Patient identifier
- **Age**: Patient age (29-77 years)
- **Sex**: Gender (0=Female, 1=Male)
- **Angina**: Chest pain type (1-4)
- **Blood_Pressure**: Resting blood pressure (94-200 mmHg)
- **Cholesterol**: Serum cholesterol (126-564 mg/dl)
- **Glycemia**: Fasting blood sugar (0=<120 mg/dl, 1=>120 mg/dl)
- **ECG**: Resting electrocardiographic results (0-2)
- **Heart_Rate**: Maximum heart rate achieved (71-202)
- **Angina_After_Sport**: Exercise-induced angina (0=No, 1=Yes)
- **ECG_Angina**: ST depression induced by exercise relative to rest
- **ECG_Slope**: Slope of peak exercise ST segment (1-3)
- **Fluoroscopy**: Number of major vessels colored by fluoroscopy (0-3)
- **Thalassaemia**: Thalassemia type (3, 6, 7)
- **Disease**: Target variable (0=No disease, 1-4=Disease presence)

## Data Preprocessing

### 1. Data Cleaning
- Replaced missing values ("?") with null values
- Dropped 6 rows containing null values
- Final dataset: 297 records

### 2. Feature Engineering
Categorical variables were transformed into human-readable labels:

**Sex**: 
- 0 → "Female"
- 1 → "Male"

**Angina**:
- 1 → "Stable angina"
- 2 → "Unstable angina"
- 3 → "Other pains"
- 4 → "Asymptomatic"

**Glycemia**:
- 0 → "Less than 120 mg/dl"
- 1 → "More than 120 mg/dl"

**ECG**:
- 0 → "Normal"
- 1 → "Anomalies"
- 2 → "Hypertrophy"

**Angina After Sport**:
- 0 → "No"
- 1 → "Yes"

**ECG Slope**:
- 1 → "Rising"
- 2 → "Stable"
- 3 → "Falling"

**Fluoroscopy**:
- 0 → "No anomaly"
- 1 → "Low"
- 2 → "Medium"
- 3 → "High"

**Thalassemia**:
- 3 → "No"
- 6 → "Thalassaemia under control"
- 7 → "Unstable Thalassaemia"

**Disease** (Target):
- 0 → "No"
- 1-4 → "Yes" (binary classification)

### 3. Machine Learning Pipeline

The ML pipeline includes the following stages:

1. **StringIndexer**: Convert categorical string features to numerical indices
   - Sex, Angina, Glycemia, ECG, Angina_After_Sport, ECG_Slope, Fluoroscopy, Thalassemia

2. **VectorAssembler**: Combine all features into a single feature vector
   - Numerical features: Age, Blood_Pressure, Cholesterol, Heart_Rate
   - Indexed categorical features: Sex_idx, Angina_idx, Glycemia_idx, ECG_idx, angina_after_sport_idx, ecg_slope_idx, fluoroscopy_idx, thalassemia_idx

3. **Classification Models**: Three different algorithms were implemented and compared

## Machine Learning Models

### 1. Logistic Regression
- **Algorithm**: Linear classification model
- **Configuration**: Default parameters
- **Performance Metrics**:
  - Accuracy: 83.12%
  - F1-score: 83.22%
  - Precision: 73.17%
  - Recall: 93.75%

### 2. Random Forest Classifier
- **Algorithm**: Ensemble method using multiple decision trees
- **Configuration**: 100 trees (numTrees=100)
- **Performance Metrics**:
  - Binary Classification Accuracy: 90.42%
  - Multiclass Accuracy: 80.52%
  - F1-score: 80.52%
  - Precision: 80.52%
  - Recall: 80.52%

### 3. Gradient Boosted Trees (GBT)
- **Algorithm**: Sequential ensemble method
- **Configuration**: 100 iterations (maxIter=100)
- **Performance Metrics**:
  - Binary Classification Accuracy: 85.35%
  - Multiclass Accuracy: 77.92%
  - F1-score: 77.92%
  - Precision: 77.92%
  - Recall: 77.92%

## Model Evaluation

### Data Split
- Training set: 70% of data
- Test set: 30% of data
- Random seed: 42 (for reproducibility)

### Evaluation Metrics
- **Binary Classification Evaluator**: Used for Random Forest and GBT models
- **Multiclass Classification Evaluator**: Used for all models with metrics including:
  - Accuracy
  - F1-score
  - Precision by label
  - Recall by label

### Confusion Matrix Analysis
The notebook includes visualization of:
- Confusion matrix heatmap
- Error distribution analysis showing prediction errors by true class

## Model Comparison Summary

| Model | Binary Accuracy | Multiclass Accuracy | F1-Score | Precision | Recall |
|-------|----------------|-------------------|----------|-----------|--------|
| **Random Forest** | **90.42%** | **80.52%** | **80.52%** | **80.52%** | **80.52%** |
| Gradient Boosted Trees | 85.35% | 77.92% | 77.92% | 77.92% | 77.92% |
| Logistic Regression | - | 83.12% | 83.22% | 73.17% | 93.75% |

## Key Findings

1. **Random Forest** achieved the highest binary classification accuracy (90.42%)
2. **Logistic Regression** showed the highest recall (93.75%) but lower precision (73.17%)
3. **Random Forest** provided the most balanced performance across all metrics
4. All models demonstrated good performance with accuracies above 77%

## Technical Implementation

### Dependencies
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
```

### Spark Session Configuration
- Application Name: "PySpark ML Pipeline"
- Default Spark configuration used

### Data Loading
- Format: CSV
- Header: True
- Schema inference: Enabled
- Input file: "/content/heart-disease-68ec37d6b52cb588200595.csv"

## Usage Instructions

1. **Environment Setup**: Install PySpark and required dependencies
2. **Data Preparation**: Load the heart disease dataset
3. **Run Notebook**: Execute cells sequentially for:
   - Data cleaning and preprocessing
   - Feature engineering
   - Model training and evaluation
   - Results visualization

## Future Improvements

1. **Hyperparameter Tuning**: Use CrossValidator or TrainValidationSplit for optimal parameters
2. **Feature Selection**: Implement feature importance analysis
3. **Model Ensemble**: Combine multiple models for improved predictions
4. **Additional Metrics**: Include ROC-AUC and confusion matrix analysis for all models
5. **Cross-validation**: Implement k-fold cross-validation for more robust evaluation

## Conclusion

This project successfully demonstrates the application of PySpark MLlib for heart disease prediction, achieving good performance across multiple classification algorithms. The Random Forest model emerged as the best performer with 90.42% binary classification accuracy, making it the recommended model for this use case.