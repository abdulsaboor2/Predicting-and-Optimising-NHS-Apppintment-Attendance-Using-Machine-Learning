# Predicting and Optimising NHS Appointment Attendance Using Machine Learning

## Project Overview and Objective

Missed medical appointments are a significant challenge for healthcare systems like the NHS, leading to wasted resources and longer waiting times. This project aims to **predict patient appointment attendance** using machine learning, and suggest how understanding these patterns can help **optimise scheduling** and reduce no-shows. By analyzing historical appointment data (combining patient demographics, scheduling details, and appointment history), the model learns patterns that distinguish between attended appointments and those where patients did not attend. 

The ultimate objective is to provide a predictive tool that healthcare providers can use to identify high no-show risk appointments in advance. This could enable targeted interventions (like reminder calls or double-booking strategies) to improve attendance rates and overall efficiency.

## Dataset

The dataset consists of three CSV files included in the `datasets/` directory:

- **appointments.csv:** Each record represents an appointment with details such as scheduling date, appointment date/time, the interval between scheduling and appointment, and the outcome status (`attended`, `did not attend`, `cancelled`, etc.). This is the primary dataset used for modeling attendance.
- **patients.csv:** Demographic information for patients (e.g., patient ID, sex, date of birth, insurance provider). These details can be linked to appointments and used as features (for example, age or sex may correlate with attendance patterns).
- **slots.csv:** Information about appointment slots (times and dates, and whether the slot was available or booked). This can be used to understand scheduling context (though in this project, the focus is primarily on whether a slot was attended or missed).

**Data preprocessing:** Before training the model, the script merges and cleans these datasets. For example, it calculates each patient's age from the date of birth, categorizes appointment status into "attended" vs "missed" (no-show), and removes or flags appointments that were cancelled or still scheduled (since those are not no-shows). It also derives useful features such as the lead time between scheduling and the appointment (`scheduling_interval`), and possibly the patient's past attendance history (how many previous appointments they missed). Categorical fields (like sex or insurance type) are encoded so that they can be used by the machine learning model.

## Setup and Installation

**Requirements:** This project uses Python 3 and common data science libraries. To set up the environment, install the following dependencies (for example, via pip):

- `pandas` – for data loading and manipulation
- `numpy` – for numerical operations
- `scikit-learn` – for machine learning algorithms and evaluation metrics
- `matplotlib` (and optionally `seaborn`) – for plotting the ROC curve and other visualizations

You can install the requirements with: 

```bash
pip install pandas numpy scikit-learn matplotlib
