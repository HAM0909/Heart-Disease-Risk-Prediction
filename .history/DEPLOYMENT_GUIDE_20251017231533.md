# Deployment Guide - Heart Disease Risk Prediction

## 🚀 Overview

This guide provides comprehensive instructions for deploying the Heart Disease Risk Prediction model in production environments. It covers various deployment scenarios, from single-machine setups to distributed cloud deployments.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development Setup](#local-development-setup)  
3. [Production Environment Setup](#production-environment-setup)
4. [Cloud Deployment Options](#cloud-deployment-options)
5. [Model Serving Strategies](#model-serving-strategies)
6. [Monitoring & Maintenance](#monitoring--maintenance)
7. [Security Considerations](#security-considerations)
8. [Performance Optimization](#performance-optimization)
9. [Troubleshooting](#troubleshooting)

---

## ✅ Prerequisites

### System Requirements

**Minimum Requirements**:
- **CPU**: 4 cores, 2.5 GHz
- **RAM**: 8 GB (16 GB recommended)
- **Storage**: 20 GB available space
- **Network**: Stable internet connection for cloud deployments

**Recommended for Production**:
- **CPU**: 8+ cores, 3.0+ GHz
- **RAM**: 32+ GB
- **Storage**: SSD with 100+ GB
- **Network**: High-bandwidth connection (1 Gbps+)

### Software Dependencies

#### Core Dependencies
```bash
# Java (Required for Spark)
Java 8 or 11 (OpenJDK or Oracle JDK)

# Python
Python 3.8+ (3.9+ recommended)

# Apache Spark
Spark 3.3+ with Hadoop 3.2+
```

#### Python Packages
```bash
# Core ML packages
pyspark==3.5.1
py4j==0.10.9.7

# Data processing
pandas>=1.3.0
numpy>=1.21.0

# Visualization (for monitoring)
matplotlib>=3.5.0
seaborn>=0.11.0

# Web framework (for REST API)
flask>=2.0.0
fastapi>=0.70.0  # Alternative to Flask

# Production utilities
gunicorn>=20.0.0  # WSGI server
uvicorn>=0.15.0   # ASGI server for FastAPI

# Monitoring
prometheus-client>=0.12.0
```

---

## 💻 Local Development Setup

### 1. Environment Configuration

#### Create Virtual Environment
```bash
# Using venv
python -m venv heart_disease_env
source heart_disease_env/bin/activate  # Linux/Mac
heart_disease_env\Scripts\activate     # Windows

# Using conda
conda create -n heart_disease python=3.9
conda activate heart_disease
```

#### Install Dependencies
```bash
pip install -r requirements.txt
```

#### requirements.txt
```
pyspark==3.5.1
pandas==1.5.3
numpy==1.24.3
matplotlib==3.7.1
seaborn==0.12.2
scikit-learn==1.2.2
flask==2.3.2
gunicorn==20.1.0
prometheus-client==0.16.0
jupyter==1.0.0
pytest==7.4.0
black==23.3.0
```

### 2. Environment Variables

#### Create .env file
```bash
# Spark Configuration
SPARK_HOME=/path/to/spark
JAVA_HOME=/path/to/java
PYSPARK_PYTHON=python
PYSPARK_DRIVER_PYTHON=python

# Application Configuration
MODEL_PATH=./models/pipeline_lr_model
DATA_PATH=./data
LOG_LEVEL=INFO
PORT=8080

# Resource Configuration
SPARK_EXECUTOR_MEMORY=2g
SPARK_DRIVER_MEMORY=1g
SPARK_EXECUTOR_CORES=2
```

### 3. Project Structure

```
heart_disease_prediction/
├── src/
│   ├── __init__.py
│   ├── data_processor.py      # Data preprocessing
│   ├── model_trainer.py       # Model training
│   ├── predictor.py          # Prediction service
│   └── utils.py              # Utility functions
├── models/                   # Trained models
├── data/                     # Training data
├── tests/                    # Unit tests
├── notebooks/                # Development notebooks
├── config/
│   ├── spark_config.py       # Spark configurations
│   └── model_config.py       # Model configurations
├── deployment/
│   ├── docker/              # Docker configurations
│   ├── kubernetes/          # K8s manifests
│   └── terraform/           # Infrastructure as code
├── monitoring/
│   ├── prometheus/          # Monitoring configs
│   └── grafana/             # Dashboard configs
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🏭 Production Environment Setup

### 1. Containerized Deployment with Docker

#### Dockerfile
```dockerfile
FROM openjdk:11-jre-slim

# Install Python
RUN apt-get update && apt-get install -y python3 python3-pip

# Set working directory
WORKDIR /app

# Install Spark
ENV SPARK_VERSION=3.5.1
ENV HADOOP_VERSION=3
RUN wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
    && tar -xzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
    && mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark \
    && rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz

# Set environment variables
ENV SPARK_HOME=/opt/spark
ENV JAVA_HOME=/usr/local/openjdk-11
ENV PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY config/ ./config/

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start command
CMD ["python3", "-m", "src.app"]
```

#### docker-compose.yml
```yaml
version: '3.8'

services:
  heart-disease-api:
    build: .
    ports:
      - "8080:8080"
    environment:
      - SPARK_EXECUTOR_MEMORY=2g
      - SPARK_DRIVER_MEMORY=1g
      - MODEL_PATH=/app/models/pipeline_lr_model
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data:ro
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus:/etc/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - ./monitoring/grafana:/var/lib/grafana
```

### 2. Flask REST API Implementation

#### src/app.py
```python
from flask import Flask, request, jsonify
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import logging
import os
import time
from prometheus_client import Counter, Histogram, generate_latest

app = Flask(__name__)

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')

class HeartDiseasePredictor:
    def __init__(self):
        self.spark = None
        self.model = None
        self.initialize()
    
    def initialize(self):
        """Initialize Spark session and load model"""
        try:
            self.spark = SparkSession.builder \
                .appName("HeartDiseaseAPI") \
                .config("spark.executor.memory", os.getenv("SPARK_EXECUTOR_MEMORY", "2g")) \
                .config("spark.driver.memory", os.getenv("SPARK_DRIVER_MEMORY", "1g")) \
                .config("spark.sql.adaptive.enabled", "true") \
                .getOrCreate()
            
            model_path = os.getenv("MODEL_PATH", "./models/pipeline_lr_model")
            self.model = PipelineModel.load(model_path)
            
            logging.info("Predictor initialized successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize predictor: {e}")
            raise
    
    def predict(self, patient_data):
        """Make prediction for patient data"""
        try:
            # Convert to DataFrame
            df = self.spark.createDataFrame([patient_data])
            
            # Make prediction
            predictions = self.model.transform(df)
            
            # Extract result
            result = predictions.select("prediction", "probability").collect()[0]
            
            return {
                "prediction": int(result.prediction),
                "probability": result.probability.toArray().tolist(),
                "risk_level": "High" if result.prediction == 1 else "Low"
            }
            
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            raise

# Global predictor instance
predictor = HeartDiseasePredictor()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    REQUEST_COUNT.labels(method='GET', endpoint='/health').inc()
    
    try:
        # Test Spark session
        predictor.spark.sql("SELECT 1").collect()
        return jsonify({"status": "healthy", "timestamp": time.time()})
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route('/predict', methods=['POST'])
@REQUEST_DURATION.time()
def predict():
    """Prediction endpoint"""
    REQUEST_COUNT.labels(method='POST', endpoint='/predict').inc()
    
    try:
        # Validate request
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        
        patient_data = request.get_json()
        
        # Validate required fields
        required_fields = [
            "Age", "Blood_Pressure", "Cholesterol", "Heart_Rate",
            "Sex", "angina", "Glycemia", "ECG", "angina_after_sport",
            "ecg_slope", "fluoroscopy", "thalassemia"
        ]
        
        missing_fields = [field for field in required_fields if field not in patient_data]
        if missing_fields:
            return jsonify({"error": f"Missing fields: {missing_fields}"}), 400
        
        # Add placeholder disease field for pipeline compatibility
        patient_data["disease"] = "No"
        
        # Make prediction
        result = predictor.predict(patient_data)
        
        return jsonify({
            "status": "success",
            "prediction": result,
            "timestamp": time.time()
        })
        
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

@app.route('/model/info', methods=['GET'])
def model_info():
    """Model information endpoint"""
    REQUEST_COUNT.labels(method='GET', endpoint='/model/info').inc()
    
    try:
        stages_info = []
        for i, stage in enumerate(predictor.model.stages):
            stages_info.append({
                "stage": i,
                "type": type(stage).__name__,
                "params": stage.extractParamMap() if hasattr(stage, 'extractParamMap') else {}
            })
        
        return jsonify({
            "model_type": "PipelineModel",
            "stages": stages_info,
            "num_stages": len(predictor.model.stages)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 8080)), debug=False)
```

#### Example API Usage

```python
import requests
import json

# Example patient data
patient = {
    "Age": 54,
    "Blood_Pressure": 130,
    "Cholesterol": 242,
    "Heart_Rate": 150,
    "Sex": "Male",
    "angina": "Asymptomatic",
    "Glycemia": "Less than 120 mg/dl",
    "ECG": "Normal",
    "angina_after_sport": "No",
    "ecg_slope": "Rising",
    "fluoroscopy": 2,
    "thalassemia": "No"
}

# Make prediction
response = requests.post(
    'http://localhost:8080/predict',
    headers={'Content-Type': 'application/json'},
    data=json.dumps(patient)
)

print(response.json())
# Output:
# {
#   "status": "success",
#   "prediction": {
#     "prediction": 1,
#     "probability": [0.175, 0.825],
#     "risk_level": "High"
#   },
#   "timestamp": 1699123456.789
# }
```

---

## ☁️ Cloud Deployment Options

### 1. Amazon Web Services (AWS)

#### Elastic Container Service (ECS)
```json
{
  "family": "heart-disease-prediction",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "heart-disease-api",
      "image": "your-repo/heart-disease-prediction:latest",
      "portMappings": [
        {
          "containerPort": 8080,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "SPARK_EXECUTOR_MEMORY",
          "value": "1g"
        },
        {
          "name": "MODEL_PATH",
          "value": "/app/models/pipeline_lr_model"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/heart-disease-prediction",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### EMR Cluster Configuration
```json
{
  "Name": "HeartDiseaseMLCluster",
  "ReleaseLabel": "emr-6.15.0",
  "Applications": [
    {"Name": "Spark"},
    {"Name": "Hadoop"}
  ],
  "ServiceRole": "EMR_DefaultRole",
  "JobFlowRole": "EMR_EC2_DefaultRole",
  "Instances": {
    "InstanceGroups": [
      {
        "Name": "Master",
        "InstanceRole": "MASTER",
        "InstanceType": "m5.xlarge",
        "InstanceCount": 1
      },
      {
        "Name": "Workers",
        "InstanceRole": "CORE",
        "InstanceType": "m5.xlarge",
        "InstanceCount": 2
      }
    ],
    "Ec2KeyName": "your-key-pair"
  }
}
```

### 2. Google Cloud Platform (GCP)

#### Cloud Run Deployment
```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: heart-disease-prediction
  annotations:
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "10"
        run.googleapis.com/cpu-throttling: "false"
        run.googleapis.com/memory: "2Gi"
        run.googleapis.com/cpu: "1000m"
    spec:
      containers:
      - image: gcr.io/your-project/heart-disease-prediction:latest
        ports:
        - containerPort: 8080
        env:
        - name: SPARK_EXECUTOR_MEMORY
          value: "1g"
        - name: MODEL_PATH
          value: "/app/models/pipeline_lr_model"
        resources:
          limits:
            cpu: "1000m"
            memory: "2Gi"
```

#### Dataproc Cluster
```bash
gcloud dataproc clusters create heart-disease-cluster \
    --enable-autoscaling \
    --max-workers=5 \
    --min-workers=2 \
    --worker-machine-type=n1-standard-4 \
    --master-machine-type=n1-standard-4 \
    --image-version=2.1-debian11 \
    --initialization-actions=gs://your-bucket/init-script.sh
```

### 3. Microsoft Azure

#### Azure Container Instances
```yaml
apiVersion: 2021-07-01
location: eastus
name: heart-disease-prediction
properties:
  containers:
  - name: heart-disease-api
    properties:
      image: your-registry.azurecr.io/heart-disease-prediction:latest
      resources:
        requests:
          cpu: 1
          memoryInGb: 2
      ports:
      - port: 8080
        protocol: TCP
      environmentVariables:
      - name: SPARK_EXECUTOR_MEMORY
        value: 1g
      - name: MODEL_PATH
        value: /app/models/pipeline_lr_model
  osType: Linux
  restartPolicy: Always
  ipAddress:
    type: Public
    ports:
    - protocol: tcp
      port: 8080
tags:
  environment: production
  application: heart-disease-prediction
```

---

## 🔧 Model Serving Strategies

### 1. Batch Prediction Service

```python
# src/batch_predictor.py
import os
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from datetime import datetime
import logging

class BatchPredictor:
    def __init__(self, model_path, input_path, output_path):
        self.model_path = model_path
        self.input_path = input_path
        self.output_path = output_path
        self.spark = None
        self.model = None
        
    def initialize(self):
        """Initialize Spark and load model"""
        self.spark = SparkSession.builder \
            .appName("HeartDiseaseBatchPredictor") \
            .config("spark.executor.memory", "4g") \
            .config("spark.driver.memory", "2g") \
            .config("spark.executor.cores", "4") \
            .getOrCreate()
        
        self.model = PipelineModel.load(self.model_path)
        
    def run_batch_prediction(self):
        """Process batch of patients"""
        try:
            # Load input data
            input_df = self.spark.read.parquet(self.input_path)
            
            # Add placeholder disease column
            input_df = input_df.withColumn("disease", F.lit("No"))
            
            # Make predictions
            predictions = self.model.transform(input_df)
            
            # Select relevant columns
            result_df = predictions.select(
                "patient_id",
                "prediction",
                "probability",
                F.current_timestamp().alias("prediction_timestamp")
            )
            
            # Write results
            result_df.write \
                .mode("overwrite") \
                .parquet(f"{self.output_path}/predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            logging.info(f"Batch prediction completed. Processed {result_df.count()} records")
            
        except Exception as e:
            logging.error(f"Batch prediction failed: {e}")
            raise
        finally:
            if self.spark:
                self.spark.stop()

# Scheduler script
if __name__ == "__main__":
    predictor = BatchPredictor(
        model_path="/models/pipeline_lr_model",
        input_path="/data/input/patients.parquet",
        output_path="/data/output"
    )
    
    predictor.initialize()
    predictor.run_batch_prediction()
```

### 2. Streaming Prediction Service

```python
# src/stream_predictor.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml import PipelineModel
import json

class StreamingPredictor:
    def __init__(self, model_path, kafka_servers, input_topic, output_topic):
        self.model_path = model_path
        self.kafka_servers = kafka_servers
        self.input_topic = input_topic
        self.output_topic = output_topic
        
    def start_streaming(self):
        """Start streaming prediction service"""
        spark = SparkSession.builder \
            .appName("HeartDiseaseStreamingPredictor") \
            .config("spark.streaming.stopGracefullyOnShutdown", "true") \
            .getOrCreate()
        
        # Load model
        model = PipelineModel.load(self.model_path)
        
        # Define input stream
        input_stream = spark \
            .readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.kafka_servers) \
            .option("subscribe", self.input_topic) \
            .option("startingOffsets", "latest") \
            .load()
        
        # Parse JSON messages
        parsed_stream = input_stream.select(
            col("key").cast("string"),
            from_json(col("value").cast("string"), input_schema).alias("data")
        ).select("key", "data.*")
        
        # Add placeholder disease column
        prepared_stream = parsed_stream.withColumn("disease", lit("No"))
        
        # Make predictions
        predictions = model.transform(prepared_stream)
        
        # Format output
        output_stream = predictions.select(
            col("key"),
            to_json(struct(
                col("patient_id"),
                col("prediction").alias("risk_prediction"),
                col("probability").cast("string").alias("risk_probability"),
                current_timestamp().alias("timestamp")
            )).alias("value")
        )
        
        # Write to output topic
        query = output_stream \
            .writeStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.kafka_servers) \
            .option("topic", self.output_topic) \
            .option("checkpointLocation", "/tmp/checkpoint") \
            .trigger(processingTime="10 seconds") \
            .start()
        
        query.awaitTermination()
```

### 3. Model Registry Integration

```python
# src/model_registry.py
import mlflow
import mlflow.spark
from mlflow.tracking import MlflowClient

class ModelRegistry:
    def __init__(self, tracking_uri, registry_uri=None):
        mlflow.set_tracking_uri(tracking_uri)
        if registry_uri:
            mlflow.set_registry_uri(registry_uri)
        self.client = MlflowClient()
    
    def register_model(self, model, model_name, run_id):
        """Register model in MLflow registry"""
        model_uri = f"runs:/{run_id}/model"
        
        # Register model
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )
        
        return registered_model
    
    def promote_model(self, model_name, version, stage):
        """Promote model to different stage"""
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
    
    def load_production_model(self, model_name):
        """Load latest production model"""
        model_version = self.client.get_latest_versions(
            model_name, stages=["Production"]
        )[0]
        
        model_uri = f"models:/{model_name}/{model_version.version}"
        return mlflow.spark.load_model(model_uri)
```

---

## 📊 Monitoring & Maintenance

### 1. Health Monitoring

#### Prometheus Configuration
```yaml
# monitoring/prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'heart-disease-api'
    static_configs:
      - targets: ['heart-disease-api:8080']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'spark-metrics'
    static_configs:
      - targets: ['heart-disease-api:4040']
    metrics_path: '/metrics/json'
    scrape_interval: 10s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

#### Alert Rules
```yaml
# monitoring/prometheus/alert_rules.yml
groups:
- name: heart-disease-api
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value }} errors per second"

  - alert: HighLatency
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High latency detected"
      description: "95th percentile latency is {{ $value }} seconds"

  - alert: ModelPredictionAccuracyDrop
    expr: model_accuracy < 0.75
    for: 10m
    labels:
      severity: critical
    annotations:
      summary: "Model accuracy dropped below threshold"
      description: "Current model accuracy is {{ $value }}"
```

### 2. Application Logging

```python
# src/logging_config.py
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        if hasattr(record, 'patient_id'):
            log_entry['patient_id'] = record.patient_id
        if hasattr(record, 'prediction_time'):
            log_entry['prediction_time'] = record.prediction_time
            
        return json.dumps(log_entry)

def setup_logging():
    """Configure application logging"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(JSONFormatter())
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler('/app/logs/application.log')
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)
    
    return logger
```

### 3. Model Performance Monitoring

```python
# src/model_monitor.py
import pandas as pd
import numpy as np
from prometheus_client import Gauge, Counter
from datetime import datetime, timedelta
import sqlite3

# Metrics
model_accuracy = Gauge('model_accuracy', 'Current model accuracy')
model_drift = Gauge('model_drift', 'Model drift score')
prediction_distribution = Counter('prediction_distribution', 'Distribution of predictions', ['prediction'])

class ModelMonitor:
    def __init__(self, db_path='model_monitoring.db'):
        self.db_path = db_path
        self.initialize_db()
    
    def initialize_db(self):
        """Initialize monitoring database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                features TEXT,
                prediction INTEGER,
                probability REAL,
                actual INTEGER NULL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                metric_name TEXT,
                metric_value REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_prediction(self, features, prediction, probability, actual=None):
        """Log prediction for monitoring"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions (timestamp, features, prediction, probability, actual)
            VALUES (?, ?, ?, ?, ?)
        ''', (datetime.now(), str(features), prediction, probability, actual))
        
        conn.commit()
        conn.close()
        
        # Update Prometheus metrics
        prediction_distribution.labels(prediction=str(prediction)).inc()
    
    def calculate_accuracy(self, days=7):
        """Calculate model accuracy over time period"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT prediction, actual
            FROM predictions
            WHERE actual IS NOT NULL
            AND timestamp >= datetime('now', '-{} days')
        '''.format(days)
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(df) > 0:
            accuracy = (df['prediction'] == df['actual']).mean()
            model_accuracy.set(accuracy)
            
            # Log metric
            self.log_metric('accuracy', accuracy)
            return accuracy
        
        return None
    
    def detect_drift(self, reference_features, current_features):
        """Detect data drift using statistical methods"""
        # Simple drift detection using KL divergence
        try:
            # Convert to numpy arrays
            ref_array = np.array(reference_features)
            cur_array = np.array(current_features)
            
            # Calculate drift score (simplified)
            drift_score = np.mean(np.abs(cur_array - ref_array))
            
            model_drift.set(drift_score)
            self.log_metric('drift_score', drift_score)
            
            return drift_score
            
        except Exception as e:
            logging.error(f"Drift detection failed: {e}")
            return None
    
    def log_metric(self, metric_name, value):
        """Log metric to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO model_metrics (timestamp, metric_name, metric_value)
            VALUES (?, ?, ?)
        ''', (datetime.now(), metric_name, value))
        
        conn.commit()
        conn.close()
```

---

## 🔒 Security Considerations

### 1. API Security

#### Authentication & Authorization
```python
# src/auth.py
from functools import wraps
from flask import request, jsonify
import jwt
import os

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        
        if not token:
            return jsonify({'error': 'No authorization token provided'}), 401
        
        try:
            # Remove 'Bearer ' prefix
            token = token.replace('Bearer ', '')
            
            # Decode JWT token
            payload = jwt.decode(
                token, 
                os.getenv('JWT_SECRET'), 
                algorithms=['HS256']
            )
            
            # Add user info to request context
            request.user = payload
            
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        
        return f(*args, **kwargs)
    
    return decorated

# Apply to prediction endpoint
@app.route('/predict', methods=['POST'])
@require_auth
@REQUEST_DURATION.time()
def predict():
    # Existing prediction code...
    pass
```

#### Input Validation
```python
# src/validation.py
from marshmallow import Schema, fields, validate, ValidationError

class PatientDataSchema(Schema):
    Age = fields.Integer(required=True, validate=validate.Range(min=0, max=120))
    Blood_Pressure = fields.Integer(required=True, validate=validate.Range(min=80, max=250))
    Cholesterol = fields.Integer(required=True, validate=validate.Range(min=100, max=600))
    Heart_Rate = fields.Integer(required=True, validate=validate.Range(min=60, max=220))
    Sex = fields.String(required=True, validate=validate.OneOf(['Male', 'Female']))
    angina = fields.String(required=True, validate=validate.OneOf([
        'Stable angina', 'Unstable angina', 'Other pains', 'Asymptomatic'
    ]))
    # Additional field validations...

def validate_patient_data(data):
    """Validate patient data input"""
    schema = PatientDataSchema()
    try:
        validated_data = schema.load(data)
        return validated_data, None
    except ValidationError as err:
        return None, err.messages
```

### 2. Data Privacy

#### Data Encryption
```python
# src/encryption.py
from cryptography.fernet import Fernet
import os
import base64

class DataEncryptor:
    def __init__(self):
        self.key = os.getenv('ENCRYPTION_KEY', Fernet.generate_key())
        self.cipher_suite = Fernet(self.key)
    
    def encrypt_data(self, data):
        """Encrypt sensitive patient data"""
        return self.cipher_suite.encrypt(data.encode()).decode()
    
    def decrypt_data(self, encrypted_data):
        """Decrypt patient data"""
        return self.cipher_suite.decrypt(encrypted_data.encode()).decode()
    
    def hash_patient_id(self, patient_id):
        """Hash patient ID for anonymization"""
        import hashlib
        return hashlib.sha256(patient_id.encode()).hexdigest()
```

#### Audit Logging
```python
# src/audit.py
import logging
from datetime import datetime
import json

class AuditLogger:
    def __init__(self):
        self.logger = logging.getLogger('audit')
        handler = logging.FileHandler('/app/logs/audit.log')
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_prediction_request(self, user_id, patient_data, result):
        """Log prediction request for audit"""
        audit_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'event': 'prediction_request',
            'user_id': user_id,
            'patient_id': patient_data.get('patient_id'),
            'prediction': result.get('prediction'),
            'risk_level': result.get('risk_level')
        }
        
        self.logger.info(json.dumps(audit_entry))
    
    def log_model_access(self, user_id, action):
        """Log model access events"""
        audit_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'event': 'model_access',
            'user_id': user_id,
            'action': action
        }
        
        self.logger.info(json.dumps(audit_entry))
```

---

## ⚡ Performance Optimization

### 1. Spark Optimization

#### Memory Management
```python
# Optimal Spark configuration for production
spark_config = {
    # Executor configuration
    "spark.executor.memory": "4g",
    "spark.executor.cores": "4",
    "spark.executor.instances": "8",
    
    # Driver configuration  
    "spark.driver.memory": "2g",
    "spark.driver.maxResultSize": "1g",
    
    # Memory management
    "spark.executor.memoryFraction": "0.8",
    "spark.executor.memoryStorageLevel": "MEMORY_AND_DISK_SER",
    
    # Network optimization
    "spark.network.timeout": "300s",
    "spark.executor.heartbeatInterval": "60s",
    
    # Adaptive query execution
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.adaptive.coalescePartitions.enabled": "true",
    "spark.sql.adaptive.skewJoin.enabled": "true",
    
    # Serialization
    "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
    "spark.kryo.registrationRequired": "false"
}
```

#### Model Loading Optimization
```python
# src/model_cache.py
import threading
from pyspark.ml import PipelineModel

class ModelCache:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance.models = {}
        return cls._instance
    
    def load_model(self, model_name, model_path):
        """Load and cache model"""
        if model_name not in self.models:
            with self._lock:
                if model_name not in self.models:
                    self.models[model_name] = PipelineModel.load(model_path)
        return self.models[model_name]
    
    def get_model(self, model_name):
        """Get cached model"""
        return self.models.get(model_name)
```

### 2. API Performance

#### Connection Pooling
```python
# src/connection_pool.py
from pyspark.sql import SparkSession
import threading
from queue import Queue

class SparkSessionPool:
    def __init__(self, pool_size=5):
        self.pool = Queue(maxsize=pool_size)
        self.pool_size = pool_size
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize Spark session pool"""
        for i in range(self.pool_size):
            session = SparkSession.builder \
                .appName(f"HeartDiseaseAPI-{i}") \
                .config("spark.executor.memory", "1g") \
                .config("spark.driver.memory", "512m") \
                .getOrCreate()
            self.pool.put(session)
    
    def get_session(self):
        """Get session from pool"""
        return self.pool.get()
    
    def return_session(self, session):
        """Return session to pool"""
        self.pool.put(session)
```

#### Async Processing
```python
# src/async_predictor.py
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

class AsyncPredictor:
    def __init__(self, model_cache, session_pool):
        self.model_cache = model_cache
        self.session_pool = session_pool
        self.executor = ThreadPoolExecutor(max_workers=10)
    
    async def predict_async(self, patient_data):
        """Asynchronous prediction"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            self._predict_sync, 
            patient_data
        )
    
    def _predict_sync(self, patient_data):
        """Synchronous prediction worker"""
        session = self.session_pool.get_session()
        try:
            model = self.model_cache.get_model('heart_disease')
            df = session.createDataFrame([patient_data])
            predictions = model.transform(df)
            result = predictions.select("prediction", "probability").collect()[0]
            return {
                "prediction": int(result.prediction),
                "probability": result.probability.toArray().tolist()
            }
        finally:
            self.session_pool.return_session(session)
```

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. Memory Issues

**Problem**: OutOfMemoryError during predictions
```
py4j.protocol.Py4JJavaError: java.lang.OutOfMemoryError: Java heap space
```

**Solutions**:
```python
# Increase driver memory
spark.conf.set("spark.driver.memory", "4g")
spark.conf.set("spark.driver.maxResultSize", "2g")

# Optimize DataFrame operations
df.cache()  # Cache frequently accessed data
df.repartition(4)  # Optimize partitioning
```

#### 2. Model Loading Issues

**Problem**: Model not found or corrupted
```
pyspark.sql.utils.AnalysisException: Path does not exist
```

**Solutions**:
```python
# Verify model path
import os
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")

# Add error handling
try:
    model = PipelineModel.load(model_path)
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    # Implement fallback strategy
```

#### 3. Performance Issues

**Problem**: Slow prediction responses

**Solutions**:
```python
# Profile your code
import cProfile
profiler = cProfile.Profile()
profiler.enable()
# Your prediction code here
profiler.disable()
profiler.print_stats()

# Optimize DataFrame operations
df.persist()  # Persist DataFrames
df.coalesce(1)  # Reduce partition count
```

### 4. Deployment Checklist

**Pre-deployment**:
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Load testing completed
- [ ] Security scan passed
- [ ] Documentation updated
- [ ] Monitoring configured
- [ ] Backup strategy in place

**Post-deployment**:
- [ ] Health checks passing
- [ ] Metrics being collected
- [ ] Logs being generated
- [ ] Performance within SLA
- [ ] Security monitoring active

This comprehensive deployment guide provides everything needed to successfully deploy the Heart Disease Risk Prediction model in production environments, from development setup to cloud deployment and ongoing maintenance.