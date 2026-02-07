# AutoML
Built an interactive AutoML web application using Streamlit and Scikit-Learn that automates preprocessing, model selection, cross-validation, and hyperparameter tuning for both classification and regression tasks, with real-time training progress, prediction capabilities, and downloadable serialized models.
# 🚀 Automated Machine Learning (AutoML) Web App

An end-to-end Automated Machine Learning system built using **Scikit-Learn + Streamlit** that automatically handles preprocessing, model selection, hyperparameter tuning, training, prediction, and model export — all from a simple web interface.

This app allows users to upload any tabular dataset and train the best performing ML model without writing code.

---

## ✨ Features

✅ Upload custom CSV dataset  
✅ Choose Classification or Regression  
✅ Automatic preprocessing  
• Missing value handling  
• Scaling  
• Encoding (OneHot + Ordinal)  
• PCA for dimensionality reduction  

✅ Multiple models evaluated with K-Fold Cross Validation  
✅ GridSearch hyperparameter tuning  
✅ Best model auto-selected  
✅ Training time display  
✅ Progress bar during training  
✅ Predict on full dataset  
✅ Manual single-row prediction  
✅ Download predictions as CSV  
✅ Download trained model as `.pkl`  
✅ Dynamic model naming (`dataset_algorithm.pkl`)  

---

## 🧠 How it Works

### 1. Preprocessing
- Numerical → Imputation + Scaling + PCA  
- Categorical (low cardinality) → OneHot Encoding  
- Categorical (high cardinality) → Ordinal Encoding  

### 2. Model Selection (GridSearchCV + KFold)

### Classification models
- Logistic Regression
- Random Forest
- SVM
- KNN
- Gradient Boosting
- AdaBoost
- Extra Trees
- SGD
- XGBoost

### Regression models
- Linear Regression
- Ridge / Lasso
- Random Forest Regressor
- Gradient Boosting Regressor
- Extra Trees Regressor
- KNN Regressor
- XGBoost Regressor

### 3. Training
- 5-Fold Cross Validation
- Best score selected automatically
- Best model saved

### 4. Output
- Predictions
- Downloadable `.pkl` model
- Downloadable prediction CSV

---

## 🖥️ Demo Workflow

1. Upload CSV
2. Select target column
3. Choose task (classification/regression)
4. Click **Train**
5. View score + training time
6. Make predictions
7. Download results or trained model

---

## 📂 Project Structure
project/
│
├── backend_automl.py # ML pipeline + training logic
├── app.py # Streamlit UI
├── requirements.txt
└── README.md
