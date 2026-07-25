"""
Telecom Customer Churn Prediction — Flask web app.

Serves templates/index.html and exposes a POST /predict endpoint that:
  1. Reads the submitted form data
  2. Encodes categorical fields using the saved LabelEncoders (model/encoders.pkl)
  3. Runs the trained model (model/customer_churn_model.pkl) to get a prediction
  4. Returns {"prediction": "...", "probability": ...} as JSON

To (re)generate model/customer_churn_model.pkl, run notebooks/train_model.py
against the Telco Customer Churn dataset (see README.md).
"""

import os
import pickle

import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "customer_churn_model.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "model", "encoders.pkl")

app = Flask(__name__)

# --- Load model artifacts at startup -----------------------------------
model = None
feature_names = None
encoders = None
load_error = None

try:
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
    model = model_data["model"]
    feature_names = model_data["features_names"]

    with open(ENCODERS_PATH, "rb") as f:
        encoders = pickle.load(f)
except FileNotFoundError as exc:
    # App can still boot (e.g. so the UI renders) but /predict will report the issue.
    load_error = str(exc)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None or encoders is None:
        return jsonify({
            "error": (
                "Model not loaded. Train the model first by running "
                "notebooks/train_model.py and placing customer_churn_model.pkl "
                f"in the model/ folder. ({load_error})"
            )
        }), 500

    try:
        form = request.form

        input_data = {
            "gender": form["gender"],
            "SeniorCitizen": int(form["SeniorCitizen"]),
            "Partner": form["Partner"],
            "Dependents": form["Dependents"],
            "tenure": int(form["tenure"]),
            "PhoneService": form["PhoneService"],
            "MultipleLines": form["MultipleLines"],
            "InternetService": form["InternetService"],
            "OnlineSecurity": form["OnlineSecurity"],
            "OnlineBackup": form["OnlineBackup"],
            "DeviceProtection": form["DeviceProtection"],
            "TechSupport": form["TechSupport"],
            "StreamingTV": form["StreamingTV"],
            "StreamingMovies": form["StreamingMovies"],
            "Contract": form["Contract"],
            "PaperlessBilling": form["PaperlessBilling"],
            "PaymentMethod": form["PaymentMethod"],
            "MonthlyCharges": float(form["MonthlyCharges"]),
            "TotalCharges": float(form["TotalCharges"]),
        }

        input_df = pd.DataFrame([input_data])

        # Encode categorical columns with the saved encoders
        for column, encoder in encoders.items():
            input_df[column] = encoder.transform(input_df[column])

        # Ensure column order matches what the model was trained on
        input_df = input_df[feature_names]

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]  # P(Churn = Yes)

        return jsonify({
            "prediction": "Churn" if prediction == 1 else "No Churn",
            "probability": round(float(probability) * 100, 2),
        })

    except KeyError as exc:
        return jsonify({"error": f"Missing field: {exc}"}), 400
    except ValueError as exc:
        return jsonify({"error": f"Invalid value: {exc}"}), 400
    except Exception as exc:  # noqa: BLE001 - surface unexpected errors to the client
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True)
