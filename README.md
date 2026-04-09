# Disease Prediction System — Full Stack

## Folder structure

```
disease_prediction/
├── backend/
│   ├── main.py              ← FastAPI app
│   ├── requirements.txt
│   ├── models/
│   │   └── disease_model.pkl   ← your trained model goes here
│   └── data/
│       ├── symptom_list.pkl         (optional — auto-built if missing)
│       ├── symptom_precaution.csv   (optional)
│       └── symptom_Description.csv  (optional)
└── frontend/
    └── index.html           ← open in any browser
```

## 1. Backend setup

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

## 2. Export your trained model (add to your notebook)

```python
import pickle, os

os.makedirs("../backend/models", exist_ok=True)

# Save model
with open("../backend/models/disease_model.pkl", "wb") as f:
    pickle.dump(clf, f)          # clf = your trained sklearn classifier

# Save symptom column names so the API builds vectors correctly
symptom_cols = list(X_train.columns)
os.makedirs("../backend/data", exist_ok=True)
with open("../backend/data/symptom_list.pkl", "wb") as f:
    pickle.dump(symptom_cols, f)
```

## 3. Frontend

Open frontend/index.html in a browser, enter the backend URL, click "Test connection", then predict.

## API endpoints

| Method | Endpoint   | Description                    |
|--------|------------|--------------------------------|
| GET    | /health    | Model status                   |
| GET    | /symptoms  | Symptom list for autocomplete  |
| POST   | /predict   | Predict disease from symptoms  |

### POST /predict

Request: `{ "symptoms": ["fever", "headache", "chills"] }`

Response includes: disease, confidence %, description, precautions, matched_symptoms, other_predictions, model_used

## Fallback

If no model.pkl is found the API uses a built-in rule-based engine automatically.

> Disclaimer: Educational purposes only. Not a substitute for professional medical advice.
<!--  -->