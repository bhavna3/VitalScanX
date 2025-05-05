import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# Load Data
data = pd.read_csv(r"C:\Users\saikeerthana\OneDrive\Desktop\projects\diabetesApp\dataset\diabetes.csv")
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Handle missing values and outliers
zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']
for column in zero_not_accepted:
    # Replace 0 with NaN
    X[column] = X[column].replace(0, np.nan)
    
    # Calculate Q1, Q3, and IQR
    Q1 = X[column].quantile(0.25)
    Q3 = X[column].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define outlier boundaries
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Replace outliers with NaN
    X[column] = X[column].mask((X[column] < lower_bound) | (X[column] > upper_bound))
    
    # Fill NaN with median
    X[column] = X[column].fillna(X[column].median())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create pipeline with SMOTE and RandomForest
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Define hyperparameters to tune
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20, None],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2],
    'classifier__class_weight': ['balanced', 'balanced_subsample']
}

# Initialize and train model with GridSearchCV
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring=['accuracy', 'roc_auc', 'precision', 'recall'],
    refit='roc_auc',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Get best model
best_model = grid_search.best_estimator_

# Make predictions and evaluate
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Print results
print("\nBest Parameters:", grid_search.best_params_)
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nROC AUC Score:", roc_auc_score(y_test, y_pred_proba))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.named_steps['classifier'].feature_importances_
})
print("\nFeature Importance:")
print(feature_importance.sort_values('importance', ascending=False))

# Ensure Directory Exists
os.makedirs('models', exist_ok=True)

# Save Model & Scaler
joblib.dump(best_model, 'models/diabetes_model.joblib')

print("\nModel saved successfully!")

# Save feature importance thresholds
thresholds = {
    'Glucose': {'min': 70, 'max': 200},
    'BloodPressure': {'min': 60, 'max': 140},
    'BMI': {'min': 18.5, 'max': 40},
    'Age': {'min': 0, 'max': 120},
    'Insulin': {'min': 0, 'max': 850},
    'SkinThickness': {'min': 0, 'max': 100},
    'DiabetesPedigreeFunction': {'min': 0, 'max': 2.5},
    'Pregnancies': {'min': 0, 'max': 20}
}

joblib.dump(thresholds, 'models/thresholds.joblib')