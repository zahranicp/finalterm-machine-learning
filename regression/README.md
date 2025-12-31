# 🎵 Music Release Year Prediction - Machine Learning Project

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-green.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

**End-to-End Machine Learning Pipeline for Predicting Song Release Years from Audio Features**

---

## 📖 Project Overview

This project implements a comprehensive **regression pipeline** to predict the release year of songs based on audio features. The dataset contains **515,131 songs** spanning from **1922 to 2011**, with **90 audio features** extracted from music signals (timbre characteristics, spectral features, etc.).

### 🎯 Objectives
- Build an end-to-end machine learning pipeline
- Compare multiple regression algorithms (ML & Deep Learning)
- Optimize model performance through hyperparameter tuning
- Achieve high prediction accuracy for music year estimation

### 📊 Key Results
| Model | RMSE (years) | MAE (years) | R² Score | Accuracy (±5 years) |
|-------|--------------|-------------|----------|---------------------|
| **XGBoost Tuned** | **8.77** | **6.16** | **0.3464** | **56.28%** |
| XGBoost (Original) | 8.79 | 6.18 | 0.3432 | 55.94% |
| Random Forest | 9.18 | 6.61 | 0.2839 | 51.23% |
| Gradient Boosting | 8.89 | 6.25 | 0.3282 | 54.67% |
| Deep Learning | 9.68 | 7.49 | 0.2045 | 47.82% |
| Ridge Regression | 9.46 | 6.76 | 0.2403 | 49.51% |

---

## 🗂️ Dataset Information

- **Total Samples**: 515,131 songs (after removing 214 duplicates)
- **Features**: 90 audio features + engineered features (106 total)
- **Target Variable**: Release year (1922-2011)
- **Data Split**: 80% Training (412,104) / 20% Testing (103,027)
- **Feature Types**: Continuous numerical values (timbre coefficients, spectral features)

### Data Source
The dataset consists of audio features extracted from music signals, where:
- **First column**: Target (Release Year)
- **Remaining 90 columns**: Audio features (feature_1 to feature_90)

---

## 🛠️ Technical Implementation

### 1️⃣ **Data Preprocessing**
- ✅ Missing values check (0 missing values)
- ✅ Duplicate removal (214 duplicates removed)
- ✅ Outlier detection (Z-score method, threshold=4)
- ✅ RobustScaler for outlier-resistant scaling

### 2️⃣ **Feature Engineering**
Created 16 additional features:
- **Polynomial Features**: Squared terms for top 5 correlated features
- **Interaction Features**: Pairwise products of top features
- **Statistical Aggregations**: Mean, std, min, max, range per feature groups
- **Feature Selection**: Removed low-variance and highly correlated features (>0.95)

### 3️⃣ **Models Implemented**

#### **Baseline Models**
1. **Linear Regression** - OLS baseline
2. **Ridge Regression** - L2 regularization (α=10.0)
3. **Lasso Regression** - L1 regularization (α=0.1)

#### **Advanced ML Models**
4. **Random Forest** - 100 trees, max_depth=20
5. **Gradient Boosting** - 100 estimators, learning_rate=0.1
6. **XGBoost** - Histogram-based, 200 trees (Original)
7. **XGBoost Tuned** - RandomizedSearchCV optimized

#### **Deep Learning**
8. **Neural Network** - 4 hidden layers (512→256→128→64)
   - Batch Normalization + Dropout
   - Adam optimizer, MSE loss
   - Early Stopping + ReduceLROnPlateau

### 4️⃣ **Hyperparameter Tuning**
- **Method**: RandomizedSearchCV
- **Iterations**: 20 combinations
- **Cross-Validation**: 3-fold
- **Scoring**: Negative MSE
- **Search Space**: 46,656 possible combinations

**Tuned Parameters (XGBoost):**
```python
{
    'n_estimators': 250,
    'max_depth': 7,
    'learning_rate': 0.1,
    'min_child_weight': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 0.5
}
```

### 5️⃣ **Evaluation Metrics**
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R²** (Coefficient of Determination)
- **Adjusted R²**
- **MAPE** (Mean Absolute Percentage Error)
- **5-Fold Cross-Validation**

---

## 📂 Repository Structure
```
music-year-prediction/
│
├── notebooks/
│   └── midterm_Regresion.ipynb          # Main Jupyter notebook
│
├── data/
│   ├── midterm-regresi-dataset.csv      # Original dataset
│   ├── train_processed_v2.csv           # Processed training data
│   └── test_processed_v2.csv            # Processed test data
│
├── models/
│   ├── xgboost_tuned.pkl                # Best model (XGBoost Tuned)
│   ├── scaler_v2.pkl                    # RobustScaler object
│   └── best_dl_model.keras              # Deep Learning model
│
├── results/
│   ├── comprehensive_results.csv        # All model results
│   ├── final_summary.csv                # Final summary report
│   ├── baseline_results.csv             # Baseline model results
│   └── visualizations/                  # All plots (.png files)
│       ├── 01_target_analysis.png
│       ├── 02_feature_correlations.png
│       ├── 03_correlation_matrix.png
│       ├── 04_outlier_detection.png
│       ├── 05_feature_importance_xgboost.png
│       ├── 06_dl_training_history.png
│       ├── 07_comprehensive_comparison.png
│       └── 08_final_evaluation.png
│
├── README.md                            # This file
└── requirements.txt                     # Python dependencies
```

---

## 🚀 How to Run

### Prerequisites
```bash
pip install pandas numpy scikit-learn xgboost tensorflow matplotlib seaborn scipy joblib
```

### Step-by-Step Execution

#### **Option 1: Run Complete Notebook**
```bash
# Open Jupyter Notebook
jupyter notebook notebooks/midterm_Regresion.ipynb

# Run all cells sequentially (Kernel → Restart & Run All)
```

#### **Option 2: Run in Google Colab**
1. Upload `midterm_Regresion.ipynb` to Google Drive
2. Open with Google Colab
3. Mount Google Drive: `drive.mount('/content/drive')`
4. Update file paths to your Drive location
5. Run all cells

#### **Option 3: Execute Individual Sections**

**1. Data Loading & EDA**
```python
# Blocks 1-3: Setup, Data Loading, EDA
# Output: Target analysis, correlations, visualizations
```

**2. Data Preprocessing**
```python
# Blocks 4-6: Quality Check, Feature Engineering, Scaling
# Output: Clean datasets saved to CSV
```

**3. Model Training**
```python
# Blocks 7-9: Baseline, Advanced ML, Deep Learning
# Output: Trained models, evaluation metrics
```

**4. Optimization & Evaluation**
```python
# Blocks 10-12: Hyperparameter Tuning, CV, Final Evaluation
# Output: Tuned model, comprehensive visualizations
```

---

## 📊 Model Performance Analysis

### Best Model: XGBoost Tuned

#### Performance Breakdown
- **R² Score**: 0.3464 (34.64% variance explained)
- **RMSE**: 8.77 years (average prediction error)
- **MAE**: 6.16 years (mean absolute error)
- **Training Time**: 28.15 minutes (with hyperparameter tuning)

#### Prediction Accuracy by Tolerance
| Tolerance | Accuracy |
|-----------|----------|
| ±1 year | 12.85% |
| ±3 years | 36.79% |
| ±5 years | 56.28% |
| ±10 years | 82.43% |

#### Cross-Validation Results (5-Fold)
- **Mean CV RMSE**: 8.85 ± 0.03 years
- **Consistency**: Low variance across folds (stable model)

### Key Insights
1. **Feature Importance**: `feature_1` (timbre coefficient) is most predictive
2. **Temporal Patterns**: Model performs better on recent years (1990-2011)
3. **Outliers**: Some extreme predictions (±71 years) for ambiguous songs
4. **Model Stability**: Tuned XGBoost shows no overfitting (Train R²=0.46 vs Test R²=0.35)

---

## 📈 Visualizations

### 1. Target Distribution
Analysis of year distribution across dataset (1922-2011)

### 2. Feature Correlations
Top positive/negative correlations with target variable

### 3. Outlier Detection
Box plots showing outlier distribution across features

### 4. Feature Importance (XGBoost)
Top 15 most predictive features

### 5. Training History (Deep Learning)
Loss and MAE curves across epochs

### 6. Model Comparison
R², RMSE, MAE, and training time comparison

### 7. Residual Analysis
- Actual vs Predicted scatter plot
- Residual distribution histogram
- Q-Q plot for normality test
- Error distribution by year range

### 8. Comprehensive Diagnostics
8-panel visualization with full model evaluation

---

## 🔬 Methodology

### Pipeline Overview
```
Raw Data (515K songs, 90 features)
    ↓
Data Cleaning (remove duplicates, check quality)
    ↓
Feature Engineering (+16 features → 106 total)
    ↓
Feature Selection (remove low-variance & correlated)
    ↓
Train-Test Split (80-20)
    ↓
Robust Scaling (outlier-resistant)
    ↓
Model Training (6 ML + 1 DL models)
    ↓
Hyperparameter Tuning (RandomizedSearchCV)
    ↓
Cross-Validation (5-fold)
    ↓
Final Evaluation & Model Selection
    ↓
Best Model: XGBoost Tuned (RMSE=8.77, R²=0.35)
```

---

## 💡 Key Learnings

### Successes ✅
- Feature engineering improved R² by ~5%
- XGBoost significantly outperformed linear models
- Hyperparameter tuning provided marginal gains (+0.32% R²)
- RobustScaler effectively handled outliers

### Challenges ⚠️
- Audio features lack interpretability (numerical coefficients)
- High feature dimensionality (90 → 106)
- Temporal trends in music production complicate prediction
- Deep Learning underperformed compared to XGBoost

### Future Improvements 🚀
1. **Advanced Feature Engineering**: Domain-specific audio features
2. **Ensemble Methods**: Stacking multiple models
3. **Time-Series Approach**: Incorporate temporal dependencies
4. **External Data**: Metadata (genre, artist, lyrics sentiment)
5. **Deep Learning Optimization**: Architecture search, more data

---

## 🛡️ Technical Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.8+ |
| **ML Frameworks** | scikit-learn 1.0+, XGBoost 1.7+ |
| **Deep Learning** | TensorFlow 2.x, Keras |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Model Persistence** | Joblib, Keras HDF5 |
| **Environment** | Jupyter Notebook, Google Colab |

---

## 📜 License

This project is for **educational purposes** as part of a Machine Learning course assignment.

---

## 👨‍💻 Author

**Zahrani Cahya Priesa**  
Computer Engineering Student  
Telkom University

**Course**: Machine Learning  
**Semester**: 7/2025  
**Assignment**: Final Exam - Regression Project

---

## 🙏 Acknowledgments

- Dataset: YearPredictionMSD (Million Song Dataset subset)
- Inspiration: Music Information Retrieval (MIR) research
- Tools: Scikit-learn, XGBoost, TensorFlow communities

---

## 📧 Contact

For questions or collaboration:
- **Email**: echazahrani1920@gmail.com
- **LinkedIn**: Zahrani Cahya Priesa

---

## ⭐ Star this Repository

If you found this project helpful, please consider giving it a star! ⭐

---

**Last Updated**: December 2025
