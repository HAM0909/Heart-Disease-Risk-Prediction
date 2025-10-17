# API Documentation - Heart Disease Risk Prediction

## 📚 Table of Contents

1. [Core Classes](#core-classes)
2. [Data Processing Functions](#data-processing-functions)
3. [Machine Learning Pipeline](#machine-learning-pipeline)
4. [Evaluation Metrics](#evaluation-metrics)
5. [Utility Functions](#utility-functions)
6. [Configuration Parameters](#configuration-parameters)
7. [Error Handling](#error-handling)

---

## 🏗️ Core Classes

### SparkSession

**Description**: Main entry point for PySpark functionality. Provides access to Spark SQL and DataFrame operations.

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("PySpark ML Pipeline").getOrCreate()
```

**Methods**:
- `builder`: Static method to create SparkSession.Builder
- `read`: Returns DataFrameReader for loading data
- `createDataFrame()`: Creates DataFrame from Python data structures

**Parameters**:
- `appName (str)`: Application name for Spark UI identification

**Returns**:
- `SparkSession`: Configured Spark session instance

---

## 📊 Data Processing Functions

### Data Loading

#### `spark.read.format()`

**Description**: Configures data source format for reading files.

```python
df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("data.csv")
```

**Parameters**:
- `format (str)`: Data source format ("csv", "parquet", "json", etc.)

**Options**:
- `header (str)`: "true" if first row contains column names
- `inferSchema (str)`: "true" to automatically detect column data types
- `delimiter (str)`: Field separator character (default: ",")
- `multiline (str)`: "true" for multi-line records
- `escape (str)`: Escape character for special characters

**Returns**:
- `DataFrame`: Loaded data as Spark DataFrame

### Data Cleaning

#### `DataFrame.replace()`

**Description**: Replaces specified values with new values across the DataFrame.

```python
df_clean = df.replace("?", None)
```

**Parameters**:
- `to_replace (str/dict)`: Value(s) to be replaced
- `value (str/None)`: New value(s) to replace with

**Returns**:
- `DataFrame`: DataFrame with replaced values

#### `DataFrame.dropna()`

**Description**: Removes rows containing null values.

```python
df_no_nulls = df.dropna()
```

**Parameters**:
- `how (str)`: "any" (default) or "all" - drop strategy
- `thresh (int)`: Minimum number of non-null values required
- `subset (list)`: Column names to consider for null checking

**Returns**:
- `DataFrame`: DataFrame with null rows removed

### Statistical Analysis

#### `DataFrame.describe()`

**Description**: Generates descriptive statistics for numerical columns.

```python
df.describe().show()
```

**Returns**:
- `DataFrame`: Statistical summary (count, mean, stddev, min, max)

#### `DataFrame.groupBy()`

**Description**: Groups DataFrame by specified columns for aggregation.

```python
grouped = df.groupBy("Sex", "disease").count()
```

**Parameters**:
- `*cols (str)`: Column names to group by

**Returns**:
- `GroupedData`: Grouped data object for aggregation operations

---

## 🔄 Feature Engineering Functions

### Categorical Transformations

#### `when()` and `col()`

**Description**: Conditional column transformations using SQL-like syntax.

```python
from pyspark.sql.functions import when, col

df = df.withColumn("Sex", when(col("Sex") == 0, "Female").otherwise("Male"))
```

**Functions**:
- `when(condition, value)`: Conditional expression
- `otherwise(value)`: Default value for unmatched conditions
- `col(column_name)`: Reference to DataFrame column

**Returns**:
- `Column`: Transformed column expression

#### `DataFrame.withColumn()`

**Description**: Adds new column or replaces existing column with transformation.

```python
df_transformed = df.withColumn("new_column", transformation_expression)
```

**Parameters**:
- `colName (str)`: Name of the column to add/replace
- `col (Column)`: Column expression for transformation

**Returns**:
- `DataFrame`: DataFrame with new/modified column

### String Indexing

#### `StringIndexer`

**Description**: Encodes string categorical features to numerical indices.

```python
from pyspark.ml.feature import StringIndexer

indexer = StringIndexer(inputCol="category", outputCol="category_idx", handleInvalid='keep')
```

**Parameters**:
- `inputCol (str)`: Input column name
- `outputCol (str)`: Output column name  
- `handleInvalid (str)`: How to handle unseen labels ("error", "skip", "keep")

**Methods**:
- `fit(dataset)`: Learns the string-to-index mapping
- `transform(dataset)`: Applies the learned mapping

### Vector Assembly

#### `VectorAssembler`

**Description**: Combines multiple feature columns into a single feature vector.

```python
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(
    inputCols=['Age', 'Blood_Pressure', 'Cholesterol', 'Sex_idx'],
    outputCol='features'
)
```

**Parameters**:
- `inputCols (list)`: List of input column names
- `outputCol (str)`: Output column name for feature vector

**Methods**:
- `transform(dataset)`: Combines specified columns into vector column
- `getInputCols()`: Returns list of input column names

---

## 🤖 Machine Learning Pipeline

### Classification Algorithms

#### `LogisticRegression`

**Description**: Binary and multiclass logistic regression classifier.

```python
from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(
    featuresCol='features',
    labelCol='label', 
    maxIter=100,
    regParam=0.01,
    elasticNetParam=0.0
)
```

**Parameters**:
- `featuresCol (str)`: Features column name (default: "features")
- `labelCol (str)`: Label column name (default: "label")
- `maxIter (int)`: Maximum iterations (default: 100)
- `regParam (float)`: Regularization parameter (default: 0.0)
- `elasticNetParam (float)`: ElasticNet mixing parameter (default: 0.0)

**Methods**:
- `fit(dataset)`: Trains the model
- `transform(dataset)`: Makes predictions

**Output Columns**:
- `prediction`: Predicted class label
- `probability`: Class probabilities vector
- `rawPrediction`: Raw prediction scores

#### `RandomForestClassifier`

**Description**: Ensemble method using multiple decision trees.

```python
from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(
    featuresCol='features',
    labelCol='label',
    numTrees=100,
    maxDepth=5,
    minInstancesPerNode=1,
    seed=42
)
```

**Parameters**:
- `featuresCol (str)`: Features column name
- `labelCol (str)`: Label column name
- `numTrees (int)`: Number of trees in forest (default: 20)
- `maxDepth (int)`: Maximum tree depth (default: 5)
- `minInstancesPerNode (int)`: Minimum instances per leaf (default: 1)
- `subsamplingRate (float)`: Fraction of data for each tree (default: 1.0)
- `seed (int)`: Random seed for reproducibility

**Properties**:
- `featureImportances`: Feature importance scores (after fitting)

#### `GBTClassifier`

**Description**: Gradient-boosted tree classifier for binary classification.

```python
from pyspark.ml.classification import GBTClassifier

gbt = GBTClassifier(
    featuresCol='features',
    labelCol='label',
    maxIter=100,
    maxDepth=5,
    stepSize=0.1,
    seed=42
)
```

**Parameters**:
- `featuresCol (str)`: Features column name
- `labelCol (str)`: Label column name
- `maxIter (int)`: Maximum iterations (default: 20)
- `maxDepth (int)`: Maximum tree depth (default: 5)
- `stepSize (float)`: Learning rate (default: 0.1)
- `minInstancesPerNode (int)`: Minimum instances per node (default: 1)

**Note**: GBT only supports binary classification in Spark ML

### Pipeline Construction

#### `Pipeline`

**Description**: Chains multiple stages (transformers and estimators) into a single workflow.

```python
from pyspark.ml import Pipeline

pipeline = Pipeline(stages=[indexer1, indexer2, assembler, classifier])
```

**Parameters**:
- `stages (list)`: List of pipeline stages in execution order

**Methods**:
- `fit(dataset)`: Fits all stages and returns PipelineModel
- `getStages()`: Returns list of pipeline stages

#### `PipelineModel`

**Description**: Fitted pipeline that can be used for predictions.

```python
# After fitting a pipeline
model = pipeline.fit(train_data)
predictions = model.transform(test_data)
```

**Methods**:
- `transform(dataset)`: Applies all pipeline stages to dataset
- `write()`: Returns MLWriter for saving model
- `load(path)`: Static method to load saved model

**Properties**:
- `stages`: List of fitted pipeline stages

---

## 📏 Evaluation Metrics

### MulticlassClassificationEvaluator

**Description**: Evaluator for multiclass classification metrics.

```python
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction", 
    metricName="accuracy"
)

accuracy = evaluator.evaluate(predictions)
```

**Parameters**:
- `labelCol (str)`: True label column name
- `predictionCol (str)`: Predicted label column name
- `metricName (str)`: Metric to evaluate

**Supported Metrics**:
- `"accuracy"`: Overall accuracy
- `"f1"`: Weighted F1-score
- `"weightedPrecision"`: Weighted precision
- `"weightedRecall"`: Weighted recall
- `"precisionByLabel"`: Precision for each label
- `"recallByLabel"`: Recall for each label

### MulticlassMetrics

**Description**: Provides comprehensive multiclass classification metrics.

```python
from pyspark.mllib.evaluation import MulticlassMetrics

# Convert predictions to RDD format
rdd = predictions.select('prediction', 'label').rdd.map(
    lambda row: (row['prediction'], row['label'])
)
metrics = MulticlassMetrics(rdd)
```

**Methods**:
- `accuracy`: Overall accuracy
- `confusionMatrix()`: Confusion matrix as DenseMatrix
- `precision(label)`: Precision for specific label
- `recall(label)`: Recall for specific label
- `fMeasure(label, beta)`: F-measure for specific label
- `labels`: Array of labels

---

## 🛠️ Utility Functions

### Data Splitting

#### `DataFrame.randomSplit()`

**Description**: Randomly splits DataFrame into multiple DataFrames.

```python
train_data, test_data = df.randomSplit([0.7, 0.3], seed=42)
```

**Parameters**:
- `weights (list)`: Relative weights for each split
- `seed (int)`: Random seed for reproducibility

**Returns**:
- `list[DataFrame]`: List of split DataFrames

### Display Functions

#### `DataFrame.show()`

**Description**: Displays DataFrame contents in tabular format.

```python
df.show(n=20, truncate=True, vertical=False)
```

**Parameters**:
- `n (int)`: Number of rows to show (default: 20)
- `truncate (bool)`: Truncate long strings (default: True)
- `vertical (bool)`: Print rows vertically (default: False)

#### `DataFrame.printSchema()`

**Description**: Prints the schema of the DataFrame in tree format.

```python
df.printSchema()
```

**Output Example**:
```
root
 |-- Age: double (nullable = true)
 |-- Sex: string (nullable = true)
 |-- Disease: integer (nullable = true)
```

### Column Operations

#### `DataFrame.select()`

**Description**: Selects specific columns from DataFrame.

```python
subset = df.select("Age", "Sex", "Disease")
```

**Parameters**:
- `*cols (str/Column)`: Column names or column expressions

**Returns**:
- `DataFrame`: DataFrame with selected columns

#### `DataFrame.filter()`

**Description**: Filters DataFrame rows based on condition.

```python
adults = df.filter(col("Age") >= 18)
errors = df.filter("label != prediction")
```

**Parameters**:
- `condition (str/Column)`: Filter condition

**Returns**:
- `DataFrame`: Filtered DataFrame

---

## ⚙️ Configuration Parameters

### Spark Configuration

#### Session Configuration

```python
spark.conf.set("spark.executor.memory", "2g")
spark.conf.set("spark.driver.memory", "1g")
spark.conf.set("spark.sql.adaptive.enabled", "true")
```

**Key Parameters**:
- `spark.executor.memory`: Executor heap memory
- `spark.driver.memory`: Driver heap memory  
- `spark.sql.adaptive.enabled`: Enable adaptive query execution
- `spark.serializer`: Serialization library
- `spark.sql.adaptive.coalescePartitions.enabled`: Enable partition coalescing

### Model Hyperparameters

#### Logistic Regression Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `maxIter` | int | 100 | Maximum iterations |
| `regParam` | float | 0.0 | Regularization parameter |
| `elasticNetParam` | float | 0.0 | ElasticNet mixing (0=L2, 1=L1) |
| `threshold` | float | 0.5 | Binary classification threshold |
| `family` | str | "auto" | Model family ("binomial", "multinomial") |

#### Random Forest Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `numTrees` | int | 20 | Number of trees |
| `maxDepth` | int | 5 | Maximum tree depth |
| `minInstancesPerNode` | int | 1 | Minimum instances per leaf |
| `maxBins` | int | 32 | Maximum bins for continuous features |
| `subsamplingRate` | float | 1.0 | Data sampling rate per tree |
| `impurity` | str | "gini" | Impurity measure ("gini", "entropy") |

#### Gradient Boosting Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `maxIter` | int | 20 | Number of boosting iterations |
| `maxDepth` | int | 5 | Maximum tree depth |
| `stepSize` | float | 0.1 | Learning rate |
| `subsamplingRate` | float | 1.0 | Data sampling rate |
| `lossType` | str | "logistic" | Loss function type |

---

## 🚨 Error Handling

### Common Exceptions

#### `AnalysisException`

**Description**: SQL analysis errors (column not found, type mismatches, etc.)

```python
from pyspark.sql.utils import AnalysisException

try:
    result = df.select("nonexistent_column")
except AnalysisException as e:
    print(f"Column not found: {e}")
```

#### `Py4JJavaError`

**Description**: Java-side errors during Spark operations.

```python
from py4j.protocol import Py4JJavaError

try:
    model = pipeline.fit(train_data)
except Py4JJavaError as e:
    print(f"Java error: {e.java_exception}")
```

#### `IllegalArgumentException`

**Description**: Invalid parameter values for ML algorithms.

```python
try:
    lr = LogisticRegression(maxIter=-1)  # Invalid negative value
except ValueError as e:
    print(f"Invalid parameter: {e}")
```

### Error Prevention Best Practices

#### Data Validation

```python
def validate_dataframe(df, required_columns):
    """
    Validates DataFrame has required columns and no nulls.
    
    Args:
        df: Input DataFrame
        required_columns: List of required column names
        
    Raises:
        ValueError: If validation fails
    """
    # Check required columns exist
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    
    # Check for null values
    null_counts = df.select([
        F.sum(F.col(c).isNull().cast("int")).alias(c) 
        for c in required_columns
    ]).collect()[0].asDict()
    
    null_columns = [col for col, count in null_counts.items() if count > 0]
    if null_columns:
        raise ValueError(f"Null values found in: {null_columns}")
```

#### Model Validation

```python
def validate_model_performance(predictions, min_accuracy=0.5):
    """
    Validates model meets minimum performance requirements.
    
    Args:
        predictions: Model predictions DataFrame
        min_accuracy: Minimum required accuracy
        
    Raises:
        ValueError: If model performance is insufficient
    """
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )
    accuracy = evaluator.evaluate(predictions)
    
    if accuracy < min_accuracy:
        raise ValueError(f"Model accuracy {accuracy:.3f} below minimum {min_accuracy}")
```

---

## 📊 Data Structures

### Spark DataTypes

#### Numerical Types

```python
from pyspark.sql.types import *

# Numeric data types
IntegerType()      # 32-bit signed integers
LongType()         # 64-bit signed integers  
FloatType()        # 32-bit floating point
DoubleType()       # 64-bit floating point
DecimalType(p, s)  # Decimal with precision p and scale s
```

#### String and Binary Types

```python
StringType()       # Variable-length character strings
BinaryType()       # Binary data
```

#### Date and Time Types

```python
DateType()         # Date values (year, month, day)
TimestampType()    # Timestamp values with timezone
```

#### Complex Types

```python
ArrayType(elementType, containsNull)    # Array of elements
MapType(keyType, valueType, valueContainsNull)  # Key-value pairs
StructType([StructField(...), ...])     # Nested structure
```

### Vector Types

#### DenseVector

**Description**: Dense representation of numerical vector.

```python
from pyspark.ml.linalg import DenseVector

vector = DenseVector([1.0, 2.0, 3.0, 4.0])
print(vector.size)      # 4
print(vector.toArray()) # [1.0, 2.0, 3.0, 4.0]
```

#### SparseVector

**Description**: Sparse representation for vectors with many zeros.

```python
from pyspark.ml.linalg import SparseVector

# SparseVector(size, indices, values)
sparse_vector = SparseVector(5, [0, 2, 4], [1.0, 3.0, 5.0])
# Represents: [1.0, 0.0, 3.0, 0.0, 5.0]
```

---

## 🔧 Advanced API Usage

### Custom Transformers

#### Creating Custom Transformer

```python
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType

class CustomScaler(Transformer, HasInputCol, HasOutputCol, 
                   DefaultParamsReadable, DefaultParamsWritable):
    
    def __init__(self, inputCol=None, outputCol=None):
        super().__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)
    
    def setParams(self, inputCol=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)
    
    def _transform(self, dataset):
        def scale_value(value):
            return (value - self.mean) / self.std
        
        scale_udf = udf(scale_value, DoubleType())
        return dataset.withColumn(self.getOutputCol(), 
                                scale_udf(dataset[self.getInputCol()]))
```

### Model Persistence

#### Saving Models

```python
# Save Pipeline
pipeline.write().overwrite().save("path/to/pipeline")

# Save PipelineModel  
model.write().overwrite().save("path/to/model")

# Save with versioning
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model.write().overwrite().save(f"models/heart_disease_model_{timestamp}")
```

#### Loading Models

```python
from pyspark.ml import Pipeline, PipelineModel

# Load Pipeline
loaded_pipeline = Pipeline.load("path/to/pipeline")

# Load PipelineModel
loaded_model = PipelineModel.load("path/to/model")

# Verify model loading
print(f"Loaded model with {len(loaded_model.stages)} stages")
```

This API documentation provides comprehensive coverage of all classes, functions, and methods used in the Heart Disease Risk Prediction project. It serves as a complete reference for developers working with the codebase and understanding the PySpark ML API.