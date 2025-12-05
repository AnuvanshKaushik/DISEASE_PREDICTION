## Disease Prediction Web App

A Streamlit-based web experience that wraps your trained TensorFlow model for symptom-based disease prediction and surfaces doctor recommendations plus a short description pulled from your dataset.

### Project layout
- `app/app.py` — Streamlit UI + inference pipeline.
- `app/requirements.txt` — Python dependencies.
- `app/data/` — place `Final_Augmented_dataset_Diseases_and_Symptoms.csv` and `Description.csv`.
- `app/models/` — place `optimized_disease_prediction_model.h5`.

### Setup
1) Install Python 3.9+ and pip.
2) From the repo root:
```
pip install -r app/requirements.txt
```
3) Put your assets under `app/`:
   - `app/data/Final_Augmented_dataset_Diseases_and_Symptoms.csv`
   - `app/data/Description.csv`
   - `app/models/optimized_disease_prediction_model.h5`

### Run locally
```
cd app
streamlit run app.py
```
Then open the URL shown in the terminal (usually http://localhost:8501).

### Notes
- The app reads symptom names from your dataset, applies the same label encoding strategy, and uses your saved `.h5` model for predictions.
- If you change file names or locations, update the paths at the top of `app/app.py`.

