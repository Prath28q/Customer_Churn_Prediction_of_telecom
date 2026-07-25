# Telecom Customer Churn Prediction

A Flask web app that predicts whether a telecom customer is likely to churn,
based on a Random Forest model trained on the
[Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

## Project structure

```
telecom-churn-prediction/
├── app.py                  # Flask web app (routes: / and /predict)
├── requirements.txt
├── model/
│   ├── encoders.pkl        # Saved LabelEncoders for categorical fields
│   └── customer_churn_model.pkl   # Trained model (generate this — see below)
├── notebooks/
│   └── train_model.py      # Data prep, EDA, model training & evaluation
├── templates/
│   └── index.html          # Form UI
└── static/
    └── style.css
```

## Setup

```bash
git clone <your-repo-url>
cd telecom-churn-prediction
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Train the model

`model/customer_churn_model.pkl` is **not** included in this repo (it's
generated from data you need to download yourself).

1. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from
   [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
   and place it in the project root (or update the path in
   `notebooks/train_model.py`).
2. Run the training script:

   ```bash
   python notebooks/train_model.py
   ```

   This will:
   - Clean the data and encode categorical columns
   - Balance classes with SMOTE
   - Cross-validate Decision Tree / Random Forest / XGBoost / SVM
   - Fit the final Random Forest
   - Save `model/encoders.pkl` and `model/customer_churn_model.pkl`

   > Note: the script currently writes these `.pkl` files to the working
   > directory it's run from — move/copy them into `model/` if needed, or
   > update the `open(...)` paths in the script to point at `model/`.

## 2. Run the app

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser, fill in the customer
details, and click **Predict Churn**.

## API

`POST /predict` — accepts form-encoded data with the same fields as the UI
form and returns:

```json
{ "prediction": "Churn", "probability": 73.42 }
```

## Notes

- `encoders.pkl` is already included in `model/` (fit on the original
  training data) — if you retrain on different/updated data, regenerate and
  replace it too, since the model and encoders must stay in sync.
- `PaymentMethod` options in the form (`Bank transfer (automatic)`,
  `Credit card (automatic)`) match the exact label strings used during
  training — don't rename them without also updating/retraining the encoder.

## License

MIT
