
from flask import Flask, render_template, request
from pathlib import Path
import pandas as pd
import joblib

# --------------------------------------------------
# Initialize Flask application
# --------------------------------------------------

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "diabetes_model.pkl"

# Features must match the order used during model training
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

# Basic validation limits to reject invalid inputs
INPUT_LIMITS = {
    "Pregnancies": (0, 30),
    "Glucose": (1, 500),
    "BloodPressure": (1, 250),
    "SkinThickness": (0, 100),
    "Insulin": (0, 3000),
    "BMI": (1, 100),
    "DiabetesPedigreeFunction": (0, 10),
    "Age": (1, 120)
}

# --------------------------------------------------
# Load trained machine learning model
# --------------------------------------------------

model = None

if MODEL_PATH.exists():
    try:
        model = joblib.load(MODEL_PATH)
        print("Diabetes model loaded successfully.")
    except Exception as error:
        print("Could not load model:", error)
else:
    print("Model file not found:", MODEL_PATH)
    print("Run python train_model.py to create the model.")


# --------------------------------------------------
# Home page
# --------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# --------------------------------------------------
# Prediction page
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    if model is None:
        return (
            "Model not found or could not be loaded. "
            "Check models/diabetes_model.pkl and run train_model.py.",
            500
        )

    input_data = {}

    try:
        # Read and validate every input
        for feature in FEATURES:
            raw_value = request.form.get(feature, "").strip()

            if not raw_value:
                return (
                    f"Missing input: {feature}. "
                    "Please return to the home page and complete all fields.",
                    400
                )

            value = float(raw_value)

            if not pd.notna(value):
                return f"Invalid value for {feature}.", 400

            minimum, maximum = INPUT_LIMITS[feature]

            if value < minimum or value > maximum:
                return (
                    f"{feature} must be between "
                    f"{minimum} and {maximum}.",
                    400
                )

            if feature in ["Pregnancies", "Age"] and not value.is_integer():
                return f"{feature} must be a whole number.", 400

            input_data[feature] = value

        # Create DataFrame in the exact training feature order
        input_df = pd.DataFrame([input_data], columns=FEATURES)

        # Generate prediction
        prediction = int(model.predict(input_df)[0])

        # Calculate model-estimated probability if supported
        risk_probability = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            classes = list(model.classes_)

            if 1 in classes:
                positive_index = classes.index(1)
                risk_probability = round(
                    float(probabilities[positive_index]) * 100, 2
                )

        # Display a neutral, educational result
        if prediction == 1:
            result = "Higher Predicted Risk"
            message = (
                "The model classified this input in its positive "
                "diabetes class. This is not a medical diagnosis."
            )
        else:
            result = "Lower Predicted Risk"
            message = (
                "The model classified this input in its negative "
                "diabetes class. This does not rule out diabetes."
            )

        return render_template(
            "result.html",
            prediction=prediction,
            result=result,
            message=message,
            risk_probability=risk_probability,
            input_data=input_data
        )

    except ValueError:
        return (
            "Please enter valid numeric values in all fields.",
            400
        )

    except Exception as error:
        app.logger.exception("Prediction failed")
        return (
            "An error occurred while generating the prediction. "
            "Check the terminal for details.",
            500
        )


# --------------------------------------------------
# Health check route
# --------------------------------------------------

@app.route("/health")
def health():
    return {
        "application": "HealthRisk AI",
        "status": "running",
        "model_loaded": model is not None
    }


# --------------------------------------------------
# Run application
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)

