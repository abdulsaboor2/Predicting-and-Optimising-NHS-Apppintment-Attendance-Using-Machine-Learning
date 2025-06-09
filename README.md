# Predicting and Optimising NHS Appointment Attendance using Machine Learning

## Project Objective

Missed medical appointments cost healthcare systems significant resources. This project uses machine learning to **predict NHS patient attendance** based on appointment history, patient demographics, and scheduling factors. With better predictions, hospitals can reduce no-shows through smart scheduling and targeted interventions.

## Dataset Overview

This project integrates three real-world datasets:

* `appointments.csv`: Appointment details (appointment_id, slot_id, scheduling_date, appointment_date, appointment_time, scheduling_interval, status, check_in_time, appointment_duration, start_time, end_time, waiting_time, patient_id, sex, age, age_group).
* `patients.csv`: Patient demographics (patient_id, name, sex, dob, insurance).
* `slots.csv`: Time slot metadata (slot_id, appointment_date, appointment_time, is_available).

Data preprocessing includes:

* Age and weekday feature extraction
* Appointment lead time calculation
* One-hot encoding
* SMOTEENN for balancing class distribution

## Machine Learning Models

This research trained and evaluated **7 different models** using a common pipeline:

* Logistic Regression
* Decision Tree
* Random Forest
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)
* LightGBM
* XGBoost

All models were evaluated on:

* Accuracy
* ROC-AUC
* F1 Score
* Precision
* Recall
* Specificity
* Confusion Matrix

 **Best Performing Model:**
`Random Forest` yielded highest accuracy (92.4%) and AUC (97.9%).

## Results

| Model               | Accuracy  | ROC-AUC   | F1 Score  | Recall | Specificity |
| ------------------- | --------- | --------- | --------- | ------ | ----------- |
| Logistic Regression | 0.747     | 0.835     | 0.573     | 0.565  | 0.826       |
| Decision Tree       | 0.831     | 0.910     | 0.710     | 0.687  | 0.893       |
| Random Forest       | **0.924** | **0.979** | **0.872** | 0.856  | 0.954       |
| LightGBM            | 0.829     | 0.909     | 0.695     | 0.650  | 0.905       |
| XGBoost             | 0.831     | 0.910     | 0.710     | 0.687  | 0.893       |
| KNN                 | 0.831     | 0.910     | 0.710     | 0.687  | 0.893       |
| SVM                 | 0.779591  | 0.857354     | 0.627792     | 0.63722  | 0.893       |

## Visuals

Plots include:

* Model metric comparison (bar and box plots)
* ROC curve (for top-performing models)

> *Note: Visuals are generated automatically by the script.*

## Requirements

```bash
pip install pandas numpy scikit-learn xgboost lightgbm imbalanced-learn matplotlib seaborn
```

## Getting Started

```bash
python your_script_name.py
```
