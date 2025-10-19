# Heart Disease Risk Prediction using PySpark ML

A machine learning project that implements a comprehensive pipeline for predicting heart disease risk using Apache Spark's MLlib. This project demonstrates data preprocessing, feature engineering, and model comparison using three different classification algorithms.

## 🚀 Features

- **Data Preprocessing**: Comprehensive cleaning and feature engineering pipeline
- **Multiple ML Models**: Logistic Regression, Random Forest, and Gradient Boosted Trees
- **Model Evaluation**: Detailed performance metrics and comparison
- **Visualization**: Confusion matrix and error analysis
- **Scalable**: Built with PySpark for handling large datasets

## 📊 Model Performance

| Model | Binary Accuracy | F1-Score | Precision | Recall |
|-------|----------------|----------|-----------|--------|
| **Random Forest** | **90.42%** | **80.52%** | **80.52%** | **80.52%** |
| Gradient Boosted Trees | 85.35% | 77.92% | 77.92% | 77.92% |
| Logistic Regression | 83.12% | 83.22% | 73.17% | 93.75% |

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd "Heart Disease Risk Prediction avec PySpark ML"
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📋 Requirements

- Python 3.8+
- Apache Spark 3.4+
- Java 8 or 11 (required for Spark)

## 🏃‍♂️ Usage

1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Open `Heart Disease Risk Prediction.ipynb`

3. Run all cells to:
   - Load and preprocess the data
   - Train multiple ML models
   - Evaluate and compare performance
   - Visualize results

## 📁 Project Structure

```
Heart Disease Risk Prediction avec PySpark ML/
├── Heart Disease Risk Prediction.ipynb  # Main notebook
├── HEART_DISEASE_PREDICTION_DOCUMENTATION.md  # Detailed documentation
├── requirements.txt                     # Python dependencies
├── README.md                           # Project overview
└── .gitignore                          # Git ignore rules
```

## 📖 Dataset

The dataset contains medical information for 297 patients with features including:
- Age, Sex, Blood Pressure, Cholesterol
- ECG measurements and heart rate
- Exercise-induced symptoms
- Fluoroscopy and Thalassemia results

## 🔬 Methodology

1. **Data Cleaning**: Handle missing values and data quality issues
2. **Feature Engineering**: Transform categorical variables to meaningful labels
3. **ML Pipeline**: StringIndexer → VectorAssembler → Classification Models
4. **Model Training**: Train and tune three different algorithms
5. **Evaluation**: Comprehensive metrics and visualization

## 🤖 Models Implemented

- **Logistic Regression**: Linear classification baseline
- **Random Forest**: Ensemble method with 100 trees
- **Gradient Boosted Trees**: Sequential ensemble with 100 iterations

## 📈 Key Findings

- Random Forest achieved the highest binary classification accuracy (90.42%)
- Logistic Regression showed excellent recall (93.75%) for medical screening
- All models demonstrated strong performance above 77% accuracy
- Feature engineering significantly improved model interpretability

## 🔮 Future Enhancements

- [ ] Hyperparameter tuning with CrossValidator
- [ ] Feature importance analysis
- [ ] Model ensemble methods
- [ ] Cross-validation implementation
- [ ] ROC-AUC analysis for all models

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or suggestions, please open an issue in this repository.