# Medical Research & Diagnosis System

A multi-agent medical AI system combining **trained ML/DL models, Agentic RAG, explainable AI, and LangGraph** to analyze medical data, retrieve medical literature, and provide evidence-backed responses.

## Features

* Medical prediction using XGBoost and CNN/EfficientNet models
* Medical research retrieval using PubMed/PMC Agentic RAG
* Explainability using Grad-CAM and feature importance
* Multi-agent orchestration using LangGraph
* Agent-to-agent communication with traceable messaging
* LLM-as-a-Judge and deterministic evaluation
* FastAPI REST API for prediction, research, and chat

## Architecture

```text
                         User
                           |
                           v
                    Supervisor Agent
                       (LangGraph)
                           |
             +-------------+-------------+
             |             |             |
             v             v             v
        Vision Agent  Research Agent  Direct Answer
             |             |
        ML/DL + XAI    PubMed + RAG
             |             |
             +-------------+-------------+
                           |
                           v
                    Evaluation Agent
                           |
                           v
                     Final Response
```

## Project Structure

```text
project/
├── agents/
│   ├── graph.py
│   └── agents.py
├── rag/
│   └── rag.py
├── models/
│   └── models.py
├── weights/
├── evaluation.py
├── api.py
├── main.py
├── requirements.txt
└── .env.example
```

## Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd project
```

### 2. Create environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For Windows:

```bash
venv\Scripts\activate
```

### 3. Configure API keys

```bash
cp .env.example .env
```

Add the required keys to `.env`:

```env
GEMINI_API_KEY=your_api_key
NCBI_API_KEY=your_api_key
```

### 4. Add trained models

Place the trained model files inside `weights/`:

```text
xgboost_heart_disease.joblib
xgboost_risk.joblib
chest_xray_model.h5
brain_mri_model.h5
skin_lesion_model.h5
IDRiD_EfficientNetB0_DR_Classifier.keras
Bone_Fracture_EfficientNetB0.keras
```

Missing models will not prevent the application from starting.

## Run

```bash
python main.py
```

Or:

```bash
uvicorn main:app --reload --port 8000
```

Open the API documentation at:

```text
http://localhost:8000/docs
```

## API

| Endpoint         | Description                   |
| ---------------- | ----------------------------- |
| `POST /chat`     | Complete multi-agent workflow |
| `POST /predict`  | Direct prediction and XAI     |
| `POST /research` | PubMed research               |
| `POST /upload`   | Upload medical image          |
| `GET /health`    | System and model status       |
| `GET /models`    | Available models              |

## Evaluation

Run the full benchmark:

```bash
python evaluation.py
```

Run a quick test:

```bash
python evaluation.py --n 3
```

Results are saved to:

```text
logs/evaluation_report.json
```

## Tech Stack

**Python, LangGraph, FastAPI, TensorFlow/Keras, XGBoost, scikit-learn, PubMed/NCBI, Gemini, Grad-CAM, Agentic RAG**

## Disclaimer

This project is a research prototype. Model predictions, generated explanations, and retrieved research should not be used as a substitute for assessment by a qualified medical professional.
