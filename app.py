"""
app.py

Run:
  python app.py

Endpoints:
  /           -> home page
  /about      -> about page
  /predict    -> GET shows form; POST returns prediction page
  /api/predict -> POST JSON API (send JSON with keys = feature names)
"""

from flask import Flask, render_template, request, jsonify
import os
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

MODEL_PATH = os.path.join('models', 'gwp.pkl')

# Load model bundle
if not os.path.exists(MODEL_PATH):
    print("Model file not found. Please run model_training.py first to create models/gwp.pkl")
    model_bundle = None
else:
    with open(MODEL_PATH, 'rb') as f:
        model_bundle = pickle.load(f)

pipeline = model_bundle['pipeline'] if model_bundle else None
meta = model_bundle['meta'] if model_bundle else None

if meta:
    FEATURE_NAMES = meta['feature_names']
    NUMERIC_COLS = set(meta['numeric_cols'])
    CATEGORICAL_COLS = set(meta['categorical_cols'])
else:
    FEATURE_NAMES = []
    NUMERIC_COLS = set()
    CATEGORICAL_COLS = set()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html', meta=meta)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if pipeline is None:
        return render_template('submit.html', prediction_text="Model not found. Train the model first by running model_training.py")

    if request.method == 'GET':
        # prepare feature specs for the form
        feature_specs = []
        for f in FEATURE_NAMES:
            ftype = 'numeric' if f in NUMERIC_COLS else 'categorical'
            feature_specs.append({'name': f, 'type': ftype})
        return render_template('predict.html', feature_specs=feature_specs)

    # POST: collect form values and predict
    form = request.form
    row = {}
    for f in FEATURE_NAMES:
        raw_val = form.get(f)
        if raw_val is None or raw_val.strip() == '':
            row[f] = np.nan
        else:
            if f in NUMERIC_COLS:
                try:
                    row[f] = float(raw_val)
                except:
                    row[f] = np.nan
            else:
                row[f] = raw_val.strip()
    X = pd.DataFrame([row], columns=FEATURE_NAMES)
    try:
        pred = pipeline.predict(X)[0]
        return render_template('submit.html', prediction_text=f'Predicted {meta["target"]}: {pred:.4f}')
    except Exception as e:
        return render_template('submit.html', prediction_text=f'Error during prediction: {e}')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if pipeline is None:
        return jsonify({'error': 'model not available'}), 400
    data = request.get_json()
    if not data:
        return jsonify({'error': 'send JSON body with feature keys'}), 400
    # ensure all required features present (missing allowed -> NaN)
    row = {}
    for f in FEATURE_NAMES:
        v = data.get(f, None)
        if v is None:
            row[f] = np.nan
        else:
            if f in NUMERIC_COLS:
                try:
                    row[f] = float(v)
                except:
                    row[f] = np.nan
            else:
                row[f] = str(v)
    X = pd.DataFrame([row], columns=FEATURE_NAMES)
    try:
        pred = pipeline.predict(X)[0]
        return jsonify({'prediction': float(pred)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
