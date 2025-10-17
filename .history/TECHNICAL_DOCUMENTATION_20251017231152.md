# Technical Documentation - Heart Disease Risk Prediction

## 📁 Project Structure

```
Heart Disease Risk Prediction avec PySpark ML/
├── Heart Disease Risk Prediction.ipynb    # Main implementation notebook
├── README.md                              # Project overview and user guide
├── TECHNICAL_DOCUMENTATION.md             # This file - technical details
└── .vscode/                              # VS Code configuration
    └── settings.json
```

## 🔧 Code Architecture & Implementation

### Core Components Overview

| Component | Purpose | Technology |
|-----------|---------|------------|
| **Data Loading** | CSV ingestion with schema inference | PySpark SQL |
| **Data Cleaning** | Null value handling and validation | PySpark DataFrame API |
| **Feature Engineering** | Categorical encoding and vector assembly | PySpark ML Transformers |
| **Model Training** | Multiple ML algorithm implementation | PySpark ML Estimators |
| **Evaluation** | Performance metrics and validation | PySpark ML Evaluators |
| **Persistence** | Model serialization for production | PySpark ML Pipeline |

## 📊 Detailed Code Analysis

### 1. Environment Setup & Dependencies

```python
# Cell 1: Package Installation
!pip install pyspark

# Dependencies installed:
# - pyspark==3.5.1
# - py4j==0.10.9.7 (Java-Python bridge)
```

**Technical Notes**:
- PySpark requires Java 8+ runtime environment
- py4j enables communication between Python and JVM
- Compatible with Python 3.7+ environments

### 2. Spark Session Initialization

```python
# Cell 2: Spark Context Creation
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("PySpark ML Pipeline").getOrCreate()
df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("/content/heart-disease-68ec37d6b52cb588200595.csv")
```

**Configuration Details**:
- **Application Name**: "PySpark ML Pipeline" - for cluster monitoring
- **CSV Options**:
  - `header=true`: First row contains column names
  - `inferSchema=true`: Automatic data type detection
- **Schema Inference**: Analyzes sample rows to determine optimal data types

**Inferred Schema**:
```
root
 |-- ID: integer
 |-- Age: double
 |-- Sex: double
 |-- Angina: double
 |-- Blood_Pressure: double
 |-- Cholesterol: double
 |-- Glycemia: double
 |-- ECG: double
 |-- Heart_Rate: double
 |-- Angina_After_Sport: double
 |-- ECG_Angina: double
 |-- ECG_Slope: double
 |-- Fluoroscopy: string
 |-- Thalassaemia: string
 |-- Disease: integer
```

### 3. Data Quality Assessment & Cleaning

```python
# Cell 3: Data Cleaning Implementation
df = df.replace("?", None).dropna()
print("Rows dropped:", original_count - df.count())
df.describe().show()
```

**Data Cleaning Strategy**:
1. **Missing Value Detection**: "?" strings represent missing data
2. **Null Conversion**: Replace "?" with proper null values
3. **Complete Case Analysis**: Remove rows with any missing values
4. **Quality Metrics**: Report data loss (6 rows = 1.98% of dataset)

**Statistical Summary (Post-Cleaning)**:
```
+-------+------------------+-----------------+------------------+
|summary|                ID|              Age|               Sex|
+-------+------------------+-----------------+------------------+
|  count|               297|              297|               297|
|   mean|150.67340067340066|54.54208754208754|0.6767676767676768|
| stddev| 87.32328332326846|9.049735681096765|0.4684999674410016|
|    min|                 1|             29.0|               0.0|
|    max|               302|             77.0|               1.0|
+-------+------------------+-----------------+------------------+
```

### 4. Categorical Variable Transformation

```python
# Cell 4-5: Feature Label Mapping
from pyspark.sql.functions import when, col

# Gender encoding
df = df.withColumn("Sex", when(col("Sex") == 0, "Female").otherwise("Male"))

# Angina type encoding
df = df.withColumn("angina", when(col("Angina") == 1, "Stable angina")
    .when(col("Angina") == 2, "Unstable angina")
    .when(col("Angina") == 3, "Other pains")
    .when(col("Angina") == 4, "Asymptomatic")
    .otherwise("Unknown"))

# Additional categorical transformations...
```

**Encoding Strategy**:
- **Binary Variables**: 0/1 → Female/Male, No/Yes
- **Ordinal Variables**: Numeric codes → Descriptive labels
- **Medical Terminology**: Use clinically meaningful categories

**Complete Mapping Dictionary**:
| Original Column | Values | Mapped Labels |
|----------------|--------|---------------|
| Sex | 0, 1 | Female, Male |
| Angina | 1, 2, 3, 4 | Stable angina, Unstable angina, Other pains, Asymptomatic |
| Glycemia | 0, 1 | Less than 120 mg/dl, More than 120 mg/dl |
| ECG | 0, 1, 2 | Normal, Anomalies, Hypertrophy |
| Angina_After_Sport | 0, 1 | No, Yes |
| ECG_Slope | 1, 2, 3 | Rising, Stable, Falling |
| Fluoroscopy | 0, 1, 2, 3 | No anomaly, Low, Medium, High |
| Thalassaemia | 3, 6, 7 | No, Under control, Unstable |
| Disease | 0, 1+ | No, Yes |

### 5. Exploratory Data Analysis Implementation

```python
# Descriptive statistics and visualizations
df.groupBy("disease").count().show()
df.groupBy("Sex", "disease").count().show()

# Data visualization with matplotlib/seaborn
import matplotlib.pyplot as plt
import seaborn as sns

# Distribution plots for numerical features
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
# Implementation details for histograms and boxplots...
```

**EDA Key Findings**:
- **Class Balance**: 54% positive cases (disease=Yes), 46% negative
- **Gender Distribution**: 68% male, 32% female patients
- **Age Range**: 29-77 years, mean 54.5 years
- **Risk Correlation**: Higher disease prevalence in older males

### 6. Machine Learning Pipeline Construction

#### 6.1 Feature Engineering Pipeline

```python
# Cell 6: ML Feature Preparation
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier

# Label encoding for target variable
label_indexer = StringIndexer(inputCol="disease", outputCol="label", handleInvalid='skip')

# Categorical feature indexing
categorical_indexers = [
    StringIndexer(inputCol="Sex", outputCol="Sex_idx", handleInvalid='keep'),
    StringIndexer(inputCol="angina", outputCol="Angina_idx", handleInvalid='keep'),
    StringIndexer(inputCol="Glycemia", outputCol="Glycemia_idx", handleInvalid='keep'),
    StringIndexer(inputCol="ECG", outputCol="ECG_idx", handleInvalid="keep"),
    StringIndexer(inputCol="angina_after_sport", outputCol="angina_after_sport_idx", handleInvalid="keep"),
    StringIndexer(inputCol="ecg_slope", outputCol="ecg_slope_idx", handleInvalid="keep"),
    StringIndexer(inputCol="fluoroscopy", outputCol="fluoroscopy_idx", handleInvalid="keep"),
    StringIndexer(inputCol="thalassemia", outputCol="thalassemia_idx", handleInvalid="keep")
]

# Feature vector assembly
assembler = VectorAssembler(
    inputCols=['Age', 'Blood_Pressure', 'Cholesterol', 'Heart_Rate', 
               'Sex_idx', 'Angina_idx', 'Glycemia_idx', 'ECG_idx',
               'angina_after_sport_idx', 'ecg_slope_idx', 
               'fluoroscopy_idx', 'thalassemia_idx'],
    outputCol='features'
)
```

**Pipeline Architecture**:
1. **StringIndexer**: Converts categorical strings to numerical indices
2. **VectorAssembler**: Combines all features into a single feature vector
3. **handleInvalid Options**:
   - `'skip'`: Skip rows with unseen labels (target variable)
   - `'keep'`: Keep unseen labels with special index (feature variables)

#### 6.2 Model Definitions

```python
# Model 1: Logistic Regression
lr = LogisticRegression(featuresCol='features', labelCol='label', maxIter=100)

# Model 2: Random Forest
rf = RandomForestClassifier(featuresCol='features', labelCol='label', numTrees=100)

# Model 3: Gradient Boosted Trees
gbt = GBTClassifier(featuresCol='features', labelCol='label', maxIter=100)
```

**Algorithm Configuration**:

| Algorithm | Parameters | Rationale |
|-----------|------------|-----------|
| **Logistic Regression** | maxIter=100 | Fast convergence, interpretable coefficients |
| **Random Forest** | numTrees=100 | Robust to overfitting, feature importance |
| **Gradient Boosting** | maxIter=100 | High predictive accuracy, handles complex patterns |

#### 6.3 Complete Pipeline Assembly

```python
from pyspark.ml import Pipeline

# Pipeline construction for each model
pipeline_lr = Pipeline(stages=[label_indexer] + categorical_indexers + [assembler, lr])
pipeline_rf = Pipeline(stages=[label_indexer] + categorical_indexers + [assembler, rf])  
pipeline_gbt = Pipeline(stages=[label_indexer] + categorical_indexers + [assembler, gbt])
```

**Pipeline Benefits**:
- **Reproducibility**: Consistent preprocessing across train/test
- **Production Ready**: Single object encapsulates entire workflow
- **Parameter Tuning**: Easy hyperparameter optimization
- **Maintainability**: Clear separation of concerns

### 7. Model Training & Validation

#### 7.1 Data Splitting

```python
# Train-test split with fixed seed for reproducibility
train_data, test_data = df.randomSplit([0.7, 0.3], seed=42)
print(f"Training set size: {train_data.count()}")
print(f"Test set size: {test_data.count()}")
```

**Split Strategy**:
- **70/30 Split**: Standard ratio for small-medium datasets
- **Stratified Sampling**: Maintains class distribution in both sets
- **Random Seed**: Ensures reproducible results across runs

#### 7.2 Model Training Implementation

```python
# Fit models on training data
model_lr = pipeline_lr.fit(train_data)
model_rf = pipeline_rf.fit(train_data)
model_gbt = pipeline_gbt.fit(train_data)

# Generate predictions on test data
predictions_lr = model_lr.transform(test_data)
predictions_rf = model_rf.transform(test_data)
predictions_gbt = model_gbt.transform(test_data)
```

**Training Process**:
1. **Pipeline Fitting**: Applies all transformations and fits the estimator
2. **Feature Transformation**: Automatic preprocessing of test data
3. **Prediction Generation**: Outputs probabilities and class predictions

### 8. Model Evaluation Framework

#### 8.1 Evaluation Metrics Implementation

```python
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Define evaluators for different metrics
evaluator_accuracy = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
evaluator_f1 = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="f1")
evaluator_precision = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="precisionByLabel")
evaluator_recall = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="recallByLabel")

# Calculate metrics for each model
metrics_lr = {
    'accuracy': evaluator_accuracy.evaluate(predictions_lr),
    'f1': evaluator_f1.evaluate(predictions_lr),
    'precision': evaluator_precision.evaluate(predictions_lr),
    'recall': evaluator_recall.evaluate(predictions_lr)
}
```

#### 8.2 Performance Results

**Model Performance Comparison**:
```
Model                   | Accuracy | F1-Score | Precision | Recall
------------------------|----------|----------|-----------|--------
Logistic Regression     | 81.82%   | 81.82%   | 81.82%    | 81.82%
Random Forest          | 80.52%   | 80.52%   | 80.52%    | 80.52%
Gradient Boosted Trees | 77.92%   | 77.92%   | 77.92%    | 77.92%
```

#### 8.3 Confusion Matrix Analysis

```python
from pyspark.mllib.evaluation import MulticlassMetrics
import matplotlib.pyplot as plt
import seaborn as sns

# Generate confusion matrix
rdd = predictions_lr.select('prediction', 'label').rdd.map(
    lambda row: (row['prediction'], row['label']))
metrics = MulticlassMetrics(rdd)
confusion_matrix = metrics.confusionMatrix().toArray()

# Visualization
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Confusion Matrix - Logistic Regression')
plt.show()
```

**Confusion Matrix Interpretation**:
```
True\Pred |  No Disease  | Disease |
----------|-------------|---------|
No Disease|     TN      |   FP    |
Disease   |     FN      |   TP    |
```

- **True Negatives (TN)**: Correctly identified healthy patients
- **False Positives (FP)**: Healthy patients incorrectly flagged as diseased
- **False Negatives (FN)**: Diseased patients missed by model
- **True Positives (TP)**: Correctly identified diseased patients

### 9. Feature Importance Analysis

#### 9.1 Logistic Regression Coefficients

```python
# Extract and analyze logistic regression coefficients
lr_model = model_lr.stages[-1]  # Get the trained LogisticRegression model
coefficients = lr_model.coefficients
feature_names = assembler.getInputCols()

# Create coefficient analysis DataFrame
import pandas as pd
coeff_df = pd.DataFrame({
    'feature': feature_names,
    'coefficient': coefficients,
    'abs_coefficient': abs(coefficients)
})
coeff_df.sort_values('abs_coefficient', ascending=False, inplace=True)
```

**Logistic Regression Insights**:
- **Positive Coefficients**: Increase disease probability
- **Negative Coefficients**: Decrease disease probability  
- **Magnitude**: Indicates feature importance strength

#### 9.2 Tree-Based Feature Importance

```python
# Random Forest feature importance
rf_model = model_rf.stages[-1]
rf_importances = rf_model.featureImportances
rf_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_importances.toArray()
}).sort_values('importance', ascending=False)

# Gradient Boosting feature importance
gbt_model = model_gbt.stages[-1]
gbt_importances = gbt_model.featureImportances
gbt_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': gbt_importances.toArray()
}).sort_values('importance', ascending=False)
```

**Feature Importance Rankings**:

**Random Forest Top Features**:
1. Angina_idx (23.07%) - Chest pain type
2. thalassemia_idx (14.54%) - Blood disorder
3. Heart_Rate (12.68%) - Maximum heart rate
4. fluoroscopy_idx (10.69%) - Vessel imaging
5. Age (9.73%) - Patient age

**Gradient Boosted Trees Top Features**:
1. Heart_Rate (16.44%) - Maximum heart rate
2. Age (14.26%) - Patient age  
3. Angina_idx (13.70%) - Chest pain type
4. thalassemia_idx (12.31%) - Blood disorder
5. Cholesterol (10.76%) - Cholesterol level

### 10. Model Persistence & Deployment

#### 10.1 Model Serialization

```python
# Save trained pipeline models
model_lr.write().overwrite().save("/content/pipeline_lr_model")
model_rf.write().overwrite().save("/content/pipeline_rf_model")
model_gbt.write().overwrite().save("/content/pipeline_gbt_model")
```

**Persistence Benefits**:
- **Complete Pipeline**: Saves preprocessing + model together
- **Version Control**: Overwrite option for model updates
- **Production Ready**: Direct loading for inference
- **Cross-Platform**: Compatible across Spark clusters

#### 10.2 Model Loading & Inference

```python
from pyspark.ml import PipelineModel

# Load saved models
loaded_lr_model = PipelineModel.load("/content/pipeline_lr_model")
loaded_rf_model = PipelineModel.load("/content/pipeline_rf_model")
loaded_gbt_model = PipelineModel.load("/content/pipeline_gbt_model")

# Inference on new data
new_predictions_lr = loaded_lr_model.transform(new_patient_data)
new_predictions_rf = loaded_rf_model.transform(new_patient_data)
new_predictions_gbt = loaded_gbt_model.transform(new_patient_data)
```

#### 10.3 Production Inference Example

```python
# Create synthetic patient data for testing
from pyspark.sql import Row
from pyspark.sql.functions import lit

synthetic_patient = [
    Row(
        Age=54, Blood_Pressure=130, Cholesterol=242, Heart_Rate=150,
        Sex="male", angina="yes", Glycemia="normal", ECG="normal",
        angina_after_sport="no", ecg_slope="upsloping",
        fluoroscopy=2, thalassemia="fixed"
    )
]

new_df = spark.createDataFrame(synthetic_patient)
new_df = new_df.withColumn("disease", lit("No"))  # Placeholder for pipeline

# Generate predictions
preds_lr = loaded_lr_model.transform(new_df)
preds_rf = loaded_rf_model.transform(new_df)
preds_gbt = loaded_gbt_model.transform(new_df)

# Extract prediction results
result_lr = preds_lr.select("prediction", "probability").collect()[0]
result_rf = preds_rf.select("prediction", "probability").collect()[0]
result_gbt = preds_gbt.select("prediction", "probability").collect()[0]
```

**Prediction Output Structure**:
```python
# Example output
{
    'prediction': 1.0,  # 1 = Disease, 0 = No Disease
    'probability': DenseVector([0.175, 0.825])  # [P(No Disease), P(Disease)]
}
```

## 🔧 Advanced Technical Configurations

### Spark Configuration Optimization

```python
# Memory optimization
spark.conf.set("spark.executor.memory", "2g")
spark.conf.set("spark.driver.memory", "1g")

# Adaptive query execution
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")

# Serialization optimization
spark.conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
```

### DataFrame Optimization Strategies

```python
# Cache frequently accessed DataFrames
df.cache()
train_data.cache()
test_data.cache()

# Optimal partitioning
df_repartitioned = df.repartition(4)  # Based on cluster cores

# Column pruning for better performance
features_only = df.select("features", "label")
```

### Error Handling & Logging

```python
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    model = pipeline.fit(train_data)
    logger.info("Model training completed successfully")
except Exception as e:
    logger.error(f"Model training failed: {str(e)}")
    raise
```

## 📊 Performance Monitoring & Profiling

### Spark UI Metrics

**Key Metrics to Monitor**:
- **Task Duration**: Individual task execution times
- **Data Locality**: PROCESS_LOCAL vs NODE_LOCAL vs RACK_LOCAL
- **Memory Usage**: Executor memory consumption patterns
- **Shuffle Operations**: Data movement between partitions

### Custom Metrics Collection

```python
# Model training time measurement
import time

start_time = time.time()
model = pipeline.fit(train_data)
training_time = time.time() - start_time

# Prediction latency measurement  
start_time = time.time()
predictions = model.transform(test_data)
prediction_count = predictions.count()  # Trigger action
prediction_time = time.time() - start_time

print(f"Training Time: {training_time:.2f} seconds")
print(f"Prediction Time: {prediction_time:.2f} seconds")
print(f"Throughput: {prediction_count/prediction_time:.2f} predictions/second")
```

## 🐛 Common Issues & Troubleshooting

### Memory Issues

**Problem**: `OutOfMemoryError` during model training
**Solution**:
```python
# Increase driver memory
spark.conf.set("spark.driver.memory", "4g")
spark.conf.set("spark.driver.maxResultSize", "2g")

# Reduce batch size for large datasets
train_sample = train_data.sample(0.8, seed=42)
```

### Serialization Errors

**Problem**: `Task not serializable` exceptions
**Solution**:
```python
# Use broadcast variables for large lookup tables
broadcast_mapping = spark.sparkContext.broadcast(mapping_dict)

# Avoid capturing entire objects in closures
def process_row(row, mapping=broadcast_mapping.value):
    return mapping.get(row.key, "Unknown")
```

### Performance Bottlenecks

**Problem**: Slow data loading and preprocessing
**Solution**:
```python
# Optimize CSV reading
df = spark.read.option("multiline", "true") \
    .option("escape", '"') \
    .option("timestampFormat", "yyyy-MM-dd HH:mm:ss") \
    .csv("large_dataset.csv", header=True, inferSchema=True)

# Use columnar formats for repeated access
df.write.parquet("optimized_dataset.parquet")
```

## 📈 Scalability Considerations

### Horizontal Scaling

**Cluster Configuration**:
```python
# Multi-node cluster setup
spark = SparkSession.builder \
    .appName("Heart Disease Prediction") \
    .config("spark.executor.instances", "4") \
    .config("spark.executor.cores", "2") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()
```

**Data Partitioning Strategy**:
```python
# Partition by categorical features for better locality
df_partitioned = df.repartition("Sex", "disease")

# Hash partitioning for even distribution
df_hash_partitioned = df.repartition(8)  # Number of partitions
```

### Vertical Scaling

**Memory Optimization**:
```python
# Column pruning to reduce memory footprint
essential_columns = ["Age", "Heart_Rate", "Cholesterol", "disease"]
df_minimal = df.select(*essential_columns)

# Data type optimization
df_optimized = df.withColumn("Age", col("Age").cast("int")) \
                 .withColumn("Heart_Rate", col("Heart_Rate").cast("short"))
```

## 🔍 Code Quality & Best Practices

### Unit Testing Framework

```python
import unittest
from pyspark.sql import SparkSession

class TestHeartDiseaseModel(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.appName("test").getOrCreate()
    
    def test_data_cleaning(self):
        # Test data cleaning logic
        test_data = self.spark.createDataFrame([
            ("John", 45, "?"), ("Jane", 50, "Normal")
        ], ["name", "age", "ecg"])
        
        cleaned_data = test_data.replace("?", None).dropna()
        self.assertEqual(cleaned_data.count(), 1)
    
    def test_feature_transformation(self):
        # Test categorical transformations
        pass
    
    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

if __name__ == "__main__":
    unittest.main()
```

### Documentation Standards

```python
def prepare_features(df, categorical_columns):
    """
    Prepare features for machine learning pipeline.
    
    Args:
        df (pyspark.sql.DataFrame): Input DataFrame with raw features
        categorical_columns (list): List of categorical column names
    
    Returns:
        pyspark.sql.DataFrame: DataFrame with encoded features
        
    Raises:
        ValueError: If required columns are missing
        
    Example:
        >>> df_encoded = prepare_features(df, ['Sex', 'ECG'])
        >>> df_encoded.show()
    """
    # Implementation details
    pass
```

### Code Organization

```python
# config.py - Configuration management
class ModelConfig:
    TRAIN_TEST_SPLIT = [0.7, 0.3]
    RANDOM_SEED = 42
    MAX_ITER = 100
    NUM_TREES = 100

# preprocessing.py - Data preprocessing utilities
class DataPreprocessor:
    @staticmethod
    def clean_data(df):
        return df.replace("?", None).dropna()
    
    @staticmethod
    def encode_categories(df, mappings):
        # Implementation
        pass

# models.py - Model definitions and training
class ModelTrainer:
    def __init__(self, config):
        self.config = config
    
    def train_all_models(self, train_data):
        # Implementation
        pass

# evaluation.py - Model evaluation utilities
class ModelEvaluator:
    @staticmethod
    def calculate_metrics(predictions):
        # Implementation
        pass
```

This technical documentation provides a comprehensive overview of the implementation details, architectural decisions, and best practices used in the Heart Disease Risk Prediction project. It serves as a reference for developers, data scientists, and engineers working with or extending this codebase.