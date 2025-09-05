# 🚖 Uber Ride Data Analytics — Bagging with Random Forests

This project applies **Bagging (Bootstrap Aggregating)** using **Random Forests** on an Uber ride dataset (`ncr_ride_bookings.csv`) to analyze ride booking patterns, cancellations, and customer/driver behavior.

It also includes:

* 🔹 Data preprocessing (date/time handling, categorical encoding, feature engineering)
* 🔹 Classification on `Booking Status` (Completed, Cancelled, Incomplete, etc.)
* 🔹 Confusion matrix visualization
* 🔹 Feature importance ranking
* 🔹 Cross-validation performance plots

---

## 📂 Dataset

The dataset used: **`ncr_ride_bookings.csv`** (150,000 rows, 21 columns).

Key columns include:

* `Booking Status` (✅ Target variable)
* `Vehicle Type`, `Payment Method`, `Driver Ratings`, `Customer Rating`, `Booking Value`, `Ride Distance`
* Date/Time, Cancellation reasons, etc.

Some high-cardinality text fields (`Pickup Location`, `Drop Location`, free-text reasons) were **dropped** to avoid memory issues.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/uber-bagging-randomforest.git
cd uber-bagging-randomforest
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the training + visualization script:

```bash
python bagging.py
```

This will:

1. Preprocess the dataset
2. Train a **Random Forest Classifier** (Bagging)
3. Print classification metrics
4. Display:

   * Confusion Matrix
   * Feature Importance chart
   * Cross-Validation Accuracy plot

---

## 📊 Visualizations

* **Confusion Matrix**
  Shows how well the model predicts ride completion/cancellation.

* **Feature Importance**
  Ranks top features influencing booking outcomes (e.g., `Payment Method`, `Driver Ratings`).

* **Cross-Validation Accuracy**
  Evaluates model stability across folds.

---

## 🧠 Techniques Used

* **Bagging (Random Forests)** for ensemble classification
* **Stratified Train-Test Split** to preserve class balance
* **One-Hot Encoding** for categorical variables
* **Data Cleaning** (handling NaN, dropping text-heavy columns)

---

## 📌 Requirements

* Python 3.10+
* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn

---

## 🚀 Future Improvements

* Handle class imbalance (SMOTE / class weights)
* Try Gradient Boosting methods (XGBoost, LightGBM)
* Deploy interactive dashboard (Streamlit/Plotly Dash)
