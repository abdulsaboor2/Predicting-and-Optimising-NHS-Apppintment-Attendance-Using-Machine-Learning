import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
def load_data():
    """Loads and merges the datasets."""
    appointments = pd.read_csv("./datasets/appointments.csv")
    patients = pd.read_csv("./datasets/patients.csv")
    slots = pd.read_csv("./datasets/slots.csv")
    # Merge datasets
    df = appointments.merge(patients, on='patient_id', suffixes=('', '_patient'))
    df = df.merge(slots, on='slot_id', suffixes=('', '_slot'))
    return df

# Preprocess dataset
def preprocess_data(df):
    """Performs feature engineering and preprocessing."""
    # Convert dates
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
    df['precise_age_group'] = pd.cut(df['precise_age'], bins=bins, labels=labels)

    # Drop rows with missing values in key columns
    df.dropna(subset=['precise_age', 'days_between', 'scheduling_interval', 'sex', 'insurance'], inplace=True)

    return df

# Encode categorical features
def encode_features(df, features):
    """Encodes categorical features and balances the dataset."""
    X = df[features]
    y = df["attended_flag"]

    # Encode categorical features
    categorical = ["precise_age_group", "sex", "insurance"]
    label_encoders = {}
    for col in categorical:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    return X, y

# Create and evaluate the model
def train_and_evaluate(X, y):
    """Trains the model and evaluates it."""
    # Balance data using SMOTEENN
    smt = SMOTEENN(random_state=42)
    X_bal, y_bal = smt.fit_resample(X, y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, stratify=y_bal, random_state=42)

    # Preprocessing
    numeric = ["scheduling_interval", "days_between", "appointment_day_of_week", "precise_age"]
    categorical = ["precise_age_group", "sex", "insurance"]
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
    ])

    # LightGBM classifier
    model = LGBMClassifier(random_state=42, class_weight='balanced')

    # Pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # Hyperparameter grid for tuning
    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [10, 20, 30],
        'classifier__learning_rate': [0.01, 0.05, 0.1]
    }

    # Grid search
    grid = GridSearchCV(pipeline, param_grid, scoring='roc_auc', cv=5, verbose=1, n_jobs=-1)
    grid.fit(X_train, y_train)

    # Evaluate the model
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    # Metrics
    print("\nBest Hyperparameters:", grid.best_params_)
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Plot ROC curve
    plot_roc_curve(y_test, y_prob)

# Plot ROC-AUC Curve
def plot_roc_curve(y_test, y_prob):
    """Plots the ROC-AUC curve."""
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

# Main execution
if __name__ == "__main__":
    df = load_data()
    df = preprocess_data(df)

    features = ["scheduling_interval", "days_between", "appointment_day_of_week",
                "precise_age", "precise_age_group", "sex", "insurance"]
    X, y = encode_features(df, features)

    train_and_evaluate(X, y)