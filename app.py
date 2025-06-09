import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.combine import SMOTEENN

import warnings
warnings.filterwarnings("ignore")

# Load data
appointments = pd.read_csv("appointments.csv")
patients = pd.read_csv("patients.csv")
slots = pd.read_csv("slots.csv")

# Merge datasets
df = appointments.merge(patients, on='patient_id', suffixes=('', '_patient'))
df = df.merge(slots, on='slot_id', suffixes=('', '_slot'))

# Convert date fields
for col in ['appointment_date', 'scheduling_date', 'dob']:
    df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)

df.dropna(subset=['appointment_date', 'scheduling_date', 'dob'], inplace=True)

# Feature engineering
df['precise_age'] = (df['appointment_date'] - df['dob']).dt.days // 365
df['days_between'] = (df['appointment_date'] - df['scheduling_date']).dt.days
df['appointment_day_of_week'] = df['appointment_date'].dt.dayofweek
df['attended_flag'] = df['status'].apply(lambda x: 1 if str(x).strip().lower() == "attended" else 0)

# Age group binning
bins = [0, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 84, 89, 94, 99]
labels = ['0-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49',
          '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80-84', '85-89', '90-94', '95-99']
df['precise_age_group'] = pd.cut(df['precise_age'], bins=bins, labels=labels, include_lowest=True)

# Drop missing
df.dropna(subset=['precise_age_group', 'precise_age', 'days_between', 'scheduling_interval', 'sex', 'insurance'], inplace=True)

# Select features
features = ["scheduling_interval", "days_between", "appointment_day_of_week",
            "precise_age", "precise_age_group", "sex", "insurance"]
X = df[features].copy()
y = df["attended_flag"]

# One-hot encode
X_encoded = pd.get_dummies(X, columns=["precise_age_group", "sex", "insurance"])

# Balance dataset
smt = SMOTEENN(random_state=42)
X_bal, y_bal = smt.fit_resample(X_encoded, y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, stratify=y_bal, random_state=42)

# Numeric columns to scale
numeric = ["scheduling_interval", "days_between", "appointment_day_of_week", "precise_age"]
numeric = [col for col in numeric if col in X_encoded.columns]

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric)
], remainder='passthrough')

# Model dictionary
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "LightGBM": LGBMClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True)
}

# Results storage
results = {}

# Train and evaluate models
for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "F1-Score": f1_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "Specificity": specificity,
        "Confusion Matrix": cm
    }

# Convert to DataFrame
results_df = pd.DataFrame(results).T
print("\n📊 Model Performance Comparison:")
print(results_df)

# Clean numeric for plotting
metrics_df = results_df.drop(columns=["Confusion Matrix"]).apply(pd.to_numeric, errors='coerce')

# Boxplot
plt.figure(figsize=(12, 8))
metrics_df.boxplot()
plt.title('Box Plot of Model Metrics', fontsize=16)
plt.ylabel('Scores')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Accuracy Barplot
plt.figure(figsize=(10, 6))
sns.barplot(x=metrics_df.index, y=metrics_df['Accuracy'], palette='viridis')
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# ROC-AUC Barplot
plt.figure(figsize=(10, 6))
sns.barplot(x=metrics_df.index, y=metrics_df['ROC-AUC'], palette='coolwarm')
plt.title('Model ROC-AUC Comparison')
plt.ylabel('ROC-AUC Score')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
