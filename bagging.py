import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report,
    ConfusionMatrixDisplay, accuracy_score
)

# =============================
# 1. Load Dataset
# =============================

file_path = r"c:\Users\kdwiv\OneDrive\Desktop\Projects\Bagging with Random forest\ncr_ride_bookings.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Dataset not found at: {file_path}")

df = pd.read_csv(file_path)
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

# =============================
# 2. Preprocessing
# =============================

# Drop useless identifiers
drop_cols = [
    'Booking ID', 'Customer ID',
    'Pickup Location', 'Drop Location',
    'Reason for cancelling by Customer',
    'Driver Cancellation Reason',
    'Incomplete Rides Reason'
]
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

# Convert Date column
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['day'] = df['Date'].dt.day
    df['month'] = df['Date'].dt.month
    df['weekday'] = df['Date'].dt.weekday
    df = df.drop('Date', axis=1)

# Convert Time column
if 'Time' in df.columns:
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    df['hour'] = df['Time'].dt.hour
    df = df.drop('Time', axis=1)

# =============================
# 3. Define Features & Target
# =============================

if 'Booking Status' not in df.columns:
    raise ValueError("Booking Status column not found in dataset!")

# Separate target before encoding
y = df['Booking Status']
X = df.drop('Booking Status', axis=1)

# Pick categorical columns with low cardinality
categorical_cols = []
for col in X.select_dtypes(include=['object']).columns:
    if X[col].nunique() < 20:
        categorical_cols.append(col)
    else:
        print(f"Dropping high-cardinality column: {col}")
        X = X.drop(columns=col)

# Encode safe categoricals
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Fill NaNs
X = X.fillna(0)

# =============================
# 4. Train-Test Split
# =============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =============================
# 5. Train Random Forest (Bagging)
# =============================

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# =============================
# 6. Predictions & Evaluation
# =============================

y_pred = rf.predict(X_test)

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=rf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf.classes_)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix - Random Forest (Bagging)")
plt.show()

# =============================
# 7. Extra Visualizations
# =============================

# Feature Importance
importances = pd.Series(rf.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(15)

plt.figure(figsize=(10,6))
sns.barplot(x=top_features.values, y=top_features.index, palette="viridis")
plt.title("Top 15 Feature Importances - Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()

# Cross-Validation Performance
scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
plt.figure(figsize=(6,4))
sns.lineplot(x=range(1,6), y=scores, marker='o')
plt.axhline(scores.mean(), color='red', linestyle='--', label=f"Mean Acc = {scores.mean():.3f}")
plt.title("Cross-Validation Accuracy per Fold")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
