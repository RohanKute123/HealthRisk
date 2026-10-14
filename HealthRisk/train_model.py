
from pathlib import Path

import joblib
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay


# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "diabetes.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "diabetes_model.pkl"
CHART_PATH = OUTPUT_DIR / "confusion_matrix.png"

# Expected columns
FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

# Check dataset
if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"\nDataset not found: {DATASET_PATH}\n"
        "Create the dataset folder and put diabetes.csv inside it."
    )

# Load dataset
df = pd.read_csv(DATASET_PATH)
df.columns = df.columns.str.strip()

if "Outcome" not in df.columns:
    raise ValueError(
        "The dataset must contain an 'Outcome' column."
    )

missing_columns = [
    column for column in FEATURES
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing dataset columns: {missing_columns}"
    )

# Clean data
df = df[FEATURES + ["Outcome"]].copy()

for column in FEATURES + ["Outcome"]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna(subset=["Outcome"])
df = df[df["Outcome"].isin([0, 1])]

if len(df) < 10 or df["Outcome"].nunique() != 2:
    raise ValueError(
        "Training requires at least 10 valid rows "
        "and both Outcome classes (0 and 1)."
    )

X = df[FEATURES]
y = df["Outcome"].astype(int)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Build model pipeline
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\nHealthRisk AI - Training Results")
print("--------------------------------")
print("Dataset rows:", len(df))
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print(f"Test accuracy: {accuracy * 100:.2f}%")

# Save model
joblib.dump(model, MODEL_PATH)

# Save confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    labels=[0, 1],
    display_labels=["Class 0", "Class 1"]
)

plt.title("HealthRisk AI - Confusion Matrix")
plt.tight_layout()
plt.savefig(CHART_PATH, dpi=150)
plt.close()

print("\nModel saved:", MODEL_PATH)
print("Chart saved:", CHART_PATH)
print("\nTraining completed successfully!")
