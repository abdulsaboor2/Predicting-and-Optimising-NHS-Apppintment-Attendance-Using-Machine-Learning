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
`Random Forest` yielded the highest accuracy (92.4%) and AUC (97.9%).

## Results

### 📊 Model Performance Comparison

| Model               | Accuracy  | ROC-AUC   | F1 Score  | Precision | Recall    | Specificity | Confusion Matrix                |
|---------------------|-----------|-----------|-----------|-----------|-----------|-------------|---------------------------------|
| Logistic Regression | 0.747     | 0.835     | 0.573     | 0.582     | 0.565     | 0.826       | [[3362, 710], [761, 988]]       |
| Decision Tree       | 0.909     | 0.889     | 0.847     | 0.858     | 0.836     | 0.941       | [[3830, 242], [286, 1463]]      |
| Random Forest       | **0.924** | **0.979** | **0.872** | **0.888** | **0.856** | **0.954**   | [[3883, 189], [252, 1497]]      |
| LightGBM            | 0.829     | 0.909     | 0.695     | 0.747     | 0.650     | 0.905       | [[3686, 386], [612, 1137]]      |
| XGBoost             | 0.831     | 0.910     | 0.710     | 0.735     | 0.687     | 0.893       | [[3638, 434], [548, 1201]]      |
| KNN                 | 0.842     | 0.899     | 0.708     | 0.795     | 0.638     | 0.929       | [[3784, 288], [633, 1116]]      |
| SVM                 | 0.780     | 0.857     | 0.628     | 0.637     | 0.619     | 0.849       | [[3456, 616], [667, 1082]]      |

## Visuals

Plots include:

* Model metric comparison (bar and box plots)
* ROC curve (for top-performing models)
* Confusion Matrix of Selected Model (Confusion Matrix and Pie Chart)


## Model metric comparison (bar and box plots)

### Model Accuracy Comparison

![h](https://github.com/user-attachments/assets/8ec9449f-b483-40c3-9ef2-36e46ad0d815)

### Box Plot of Model Metrix

![download (8)](https://github.com/user-attachments/assets/ec703d9a-0b3a-41d2-9fdb-1d49d2c05ceb)

## ROC Curve

### Model ROC-AUC Comparison

![n](https://github.com/user-attachments/assets/41fc9214-57fa-40c5-8859-6a6da8dcee9e)

## Confusion Matrix of Selected Model

### Confusion Matrix of Random Forest

![download (7)](https://github.com/user-attachments/assets/c464bc0e-d0cc-4458-be72-308f6af03b95)

### Pie Chart of Correct and Incorrect Predictions of Random Forest
![download (5)](https://github.com/user-attachments/assets/2577783c-3494-42e7-beb7-b625325dc7bd)


## Requirements

```bash
pip install pandas numpy scikit-learn xgboost lightgbm imbalanced-learn matplotlib seaborn
```

## Getting Started

```bash
python your_script_name.py
```
