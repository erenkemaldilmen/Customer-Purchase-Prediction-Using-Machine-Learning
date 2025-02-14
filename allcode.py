"""
Customer Purchase Prediction Using Machine Learning
MSc in Artificial Intelligence - Machine Learning Fundamentals
Assignment: Practical Skills Assessment
"""

# Load the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.ion()  # Enable interactive plotting

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ------------------------
# 1. Data Collection
# ------------------------
# Data is read from a CSV file. Adjust the file path according to your project.
data = pd.read_csv('customer_purchase_data.csv')

# Display the first 5 rows of the dataset
print("First 5 Rows of the Dataset:")
print(data.head())

# Information about the dataset
print("\nDataset Information:")
print(data.info())

# ------------------------
# 2. Exploratory Data Analysis (EDA)
# ------------------------
# Distribution of the target variable (PurchaseStatus)
plt.figure(figsize=(6,4))
sns.countplot(x='PurchaseStatus', data=data, palette='viridis')
plt.title("PurchaseStatus Distribution")
plt.xlabel("PurchaseStatus (0: No Purchase, 1: Purchase)")
plt.ylabel("Count")
plt.show()

# Let's plot the histograms of the numerical features
numeric_features = ['Age', 'AnnualIncome', 'NumberOfPurchases', 'TimeSpentOnWebsite', 'DiscountsAvailed']
data[numeric_features].hist(bins=20, figsize=(12,8))
plt.suptitle("Distribution of Numerical Features")
plt.show()

# Check for outliers using a boxplot
plt.figure(figsize=(10,6))
sns.boxplot(data=data[numeric_features], palette='Set2')
plt.title("Boxplot of Numerical Features")
plt.xticks(rotation=45)
plt.show()

# ------------------------
# 3. Data Cleaning and Preprocessing
# ------------------------
# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# If there are missing values, drop the rows
data_clean = data.dropna().copy()

# Check for duplicate records
print("\nNumber of Duplicate Records:", data_clean.duplicated().sum())
data_clean = data_clean.drop_duplicates()

# Define the numerical and categorical columns:
numeric_cols = ['Age', 'AnnualIncome', 'NumberOfPurchases', 'TimeSpentOnWebsite', 'DiscountsAvailed']
categorical_cols = ['Gender', 'ProductCategory', 'LoyaltyProgram']

# Preprocessing: Numerical data will be scaled with StandardScaler and categorical data will be encoded with OneHotEncoder.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

# Separate the features (X) and the target variable (y):
X = data_clean.drop('PurchaseStatus', axis=1)
y = data_clean['PurchaseStatus']

# ------------------------
# 4. Splitting into Training and Test Sets
# ------------------------
# Splitting the data using a stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ------------------------
# 5. Model Selection, Design, and Training
# ------------------------
# We are using Random Forest; why? Because it is successful in capturing complex interactions,
# reduces the risk of overfitting, and allows for interpretation of feature importances.
# We combine preprocessing and model training using a Pipeline:
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Using GridSearchCV for hyperparameter tuning:
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 5, 10],
    'classifier__min_samples_split': [2, 5]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("\nBest Hyperparameters:")
print(grid_search.best_params_)

# Obtain the retrained pipeline with the best model:
best_model = grid_search.best_estimator_

# ------------------------
# 6. Model Evaluation
# ------------------------
# Make predictions on the test set:
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

# Calculate performance metrics:
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
print("\nTest Set Accuracy:", accuracy)
print("Test Set ROC AUC Score:", roc_auc)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visualization of the Confusion Matrix:
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title("Confusion Matrix")
plt.show()

# Plot the ROC curve:
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# ------------------------
# 7. Examination of Feature Importances
# ------------------------
# Retrieve the Random Forest model from the pipeline:
rf_model = best_model.named_steps['classifier']

# Obtain the categorical feature names from the one-hot encoder:
ohe = best_model.named_steps['preprocessor'].named_transformers_['cat']
cat_feature_names = ohe.get_feature_names_out(categorical_cols)
# Final feature names: first numeric, then one-hot encoded categorical features:
feature_names = numeric_cols + list(cat_feature_names)

# Calculate feature importances and store them in a DataFrame:
importances = rf_model.feature_importances_
feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)
print("\nFeature Importances:")
print(feat_imp_df)

# Visualization of feature importances:
plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette='viridis')
plt.title("Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

# ------------------------
# 8. Conclusion and Future Recommendations
# ------------------------
