# Employee Performance Prediction

## Quickstart

1. Create & activate conda env:
   conda create -n emp_perf python=3.10 -y
   conda activate emp_perf

2. Install requirements:
   pip install -r requirements.txt

3. Place your dataset file (garments_worker_productivity.csv) into `data/`

4. Train:
   python model_training.py --data data/garments_worker_productivity.csv --target target

   (replace --target with the actual column name of the column you want to predict)

5. Run the app:
   python app.py

6. Open http://127.0.0.1:5000

## Files
- model_training.py — trains models + saves the best pipeline to models/gwp.pkl
- app.py — Flask app that loads saved pipeline and serves UI for prediction
- templates/ — HTML pages
- static/style.css — basic styling
