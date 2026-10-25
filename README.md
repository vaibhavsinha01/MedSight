# Medical Model Studio

A single Streamlit app for three local models—chest X-ray, heart disease, and lifestyle risk—with explainability and an opt-in PubMed/Gemini research agent.

> Research and education only. Model outputs are not diagnoses or treatment advice. The lifestyle-risk dataset is synthetic, so test performance does not establish clinical validity.

## Setup and run

Use Python 3.11 (TensorFlow 2.16):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`. The Dockerfile runs the same Streamlit app.

## Model files

Place these files in `weights/`:

| Model | Files |
|---|---|
| Chest X-ray | `chest_xray_model.h5` |
| Heart disease | `xgboost_heart_disease.joblib` |
| Risk | `xgboost_risk.joblib` and `xgboost_risk_scaler.joblib` |

The risk model and scaler are a required pair. See `weights/README.md`.

## Retrain risk model

Run the final section of `Medical_research_agent.ipynb`. It downloads the latest [Kaggle Lifestyle and Health Risk Prediction dataset](https://www.kaggle.com/datasets/miadul/lifestyle-and-health-risk-prediction), evaluates a held-out split, then saves the model and scaler in `weights/`. Kaggle access is required to rerun the cell.

The saved model was retrained on 5,000 rows and reached **98.5% held-out accuracy** with a fixed split. Treat this only as a synthetic-data benchmark.

## Explainability

- Chest X-ray: Grad-CAM, saliency, and occlusion sensitivity maps.
- Tabular models: local SHAP and LIME feature attributions.

These methods describe model behavior; they do not prove causation or clinical findings. LIME uses a generated neighborhood because training data is not bundled.

## Research agent and privacy

The Research agent retrieves up to four relevance-ranked PubMed abstracts and uses Gemini `gemini-3.5-flash-lite` to write a concise, source-grounded summary with numbered citations. It runs only after the user checks the consent box and clicks **Search and summarize**. Set `GEMINI_API_KEY` in the git-ignored `.env` file; `NCBI_API_KEY` is optional. Without a Gemini key, the app returns the retrieved citations without a generated summary. Do not submit identifying patient information.
