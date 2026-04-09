"""
Disease Prediction System - FastAPI Backend
==========================================
Tech stack matches your repo: FastAPI + scikit-learn + pandas
Run: uvicorn main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import List, Optional
import pickle, os, re
import numpy as np
import pandas as pd

app = FastAPI(
    title="Disease Prediction API",
    description="AI-powered symptom-based disease prediction",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── PATHS ───────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "disease_model.pkl")
DATA_DIR   = os.path.join(BASE_DIR, "data")

# ─── GLOBALS ─────────────────────────────────────────────────────────────────
model = None
symptom_list: List[str] = []
precaution_df  = None
description_df = None


def load_assets():
    global model, symptom_list, precaution_df, description_df

    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("✅ ML model loaded.")
    else:
        print("⚠️  No model file — using rule-based fallback.")

    sym_file = os.path.join(DATA_DIR, "symptom_list.pkl")
    if os.path.exists(sym_file):
        with open(sym_file, "rb") as f:
            symptom_list.extend(pickle.load(f))
    else:
        raw = [
            "itching","skin_rash","nodal_skin_eruptions","continuous_sneezing","shivering",
            "chills","joint_pain","stomach_pain","acidity","ulcers_on_tongue","muscle_wasting",
            "vomiting","burning_micturition","fatigue","weight_gain","anxiety",
            "cold_hands_and_feets","mood_swings","weight_loss","restlessness","lethargy",
            "patches_in_throat","irregular_sugar_level","cough","high_fever","sunken_eyes",
            "breathlessness","sweating","dehydration","indigestion","headache","yellowish_skin",
            "dark_urine","nausea","loss_of_appetite","pain_behind_the_eyes","back_pain",
            "constipation","abdominal_pain","diarrhoea","mild_fever","yellow_urine",
            "yellowing_of_eyes","acute_liver_failure","swelling_of_stomach","swelled_lymph_nodes",
            "malaise","blurred_and_distorted_vision","phlegm","throat_irritation","redness_of_eyes",
            "sinus_pressure","runny_nose","congestion","chest_pain","weakness_in_limbs",
            "fast_heart_rate","neck_stiffness","depression","irritability","muscle_pain",
            "red_spots_over_body","belly_pain","abnormal_menstruation","increased_appetite",
            "polyuria","family_history","mucoid_sputum","rusty_sputum","lack_of_concentration",
            "visual_disturbances","blood_in_sputum","palpitations","painful_walking",
            "pus_filled_pimples","blackheads","skin_peeling","small_dents_in_nails",
            "inflammatory_nails","blister","dizziness","muscle_weakness","loss_of_balance",
            "unsteadiness","loss_of_smell","bladder_discomfort","passage_of_gases",
            "excessive_hunger","slurred_speech","knee_pain","hip_joint_pain","stiff_neck",
            "swelling_joints","movement_stiffness","spinning_movements","pale_skin",
            "sore_throat","nasal_congestion","body_ache","red_skin","spotting_urination",
        ]
        seen = set()
        for s in raw:
            if s not in seen:
                seen.add(s)
                symptom_list.append(s)

    for fname, attr in [
        ("symptom_precaution.csv", "precaution_df"),
        ("symptom_Description.csv", "description_df"),
    ]:
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            globals()[attr] = pd.read_csv(path)
            print(f"✅ Loaded {fname}")


load_assets()


# ─── NLP HELPERS ─────────────────────────────────────────────────────────────

def normalise(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s_]", "", text)
    text = re.sub(r"\s+", "_", text)
    return text


def map_symptoms(raw_inputs: List[str]) -> List[str]:
    matched = []
    for raw in raw_inputs:
        norm = normalise(raw)
        if norm in symptom_list:
            matched.append(norm); continue
        found = next((s for s in symptom_list if norm in s or s in norm), None)
        if found:
            matched.append(found); continue
        user_tok = {t for t in norm.split("_") if len(t) >= 4}
        best, best_sc = None, 0
        for sym in symptom_list:
            sc = len(user_tok & {t for t in sym.split("_") if len(t) >= 4})
            if sc > best_sc:
                best_sc, best = sc, sym
        if best_sc > 0:
            matched.append(best)
    seen = set()
    return [s for s in matched if not (s in seen or seen.add(s))]


def build_vector(matched: List[str]) -> np.ndarray:
    vec = np.zeros(len(symptom_list), dtype=int)
    for s in matched:
        if s in symptom_list:
            vec[symptom_list.index(s)] = 1
    return vec.reshape(1, -1)


# ─── RULE-BASED FALLBACK ─────────────────────────────────────────────────────
FALLBACK = {
    "Common Cold": {
        "triggers": {"runny_nose","sore_throat","cough","nasal_congestion","headache","continuous_sneezing","mild_fever"},
        "precautions": ["Rest well","Stay hydrated","Avoid cold exposure","Wash hands frequently"],
        "description": "A viral infection of the upper respiratory tract. Usually harmless and self-limiting.",
    },
    "Influenza": {
        "triggers": {"high_fever","body_ache","fatigue","chills","headache","cough","sweating"},
        "precautions": ["Isolate yourself","Rest adequately","Monitor temperature","Antiviral if prescribed"],
        "description": "A contagious respiratory illness caused by influenza viruses.",
    },
    "Typhoid Fever": {
        "triggers": {"high_fever","headache","abdominal_pain","fatigue","loss_of_appetite","nausea","constipation"},
        "precautions": ["Drink boiled water","Avoid raw food","Strict hygiene","Complete antibiotic course"],
        "description": "A bacterial infection causing sustained fever, stomach complaints, and general weakness.",
    },
    "Dengue Fever": {
        "triggers": {"high_fever","skin_rash","joint_pain","muscle_pain","headache","fatigue","pain_behind_the_eyes"},
        "precautions": ["Avoid mosquito bites","Use mosquito repellent","Monitor platelet count","Stay well hydrated"],
        "description": "A mosquito-borne viral disease causing flu-like illness.",
    },
    "Malaria": {
        "triggers": {"high_fever","chills","sweating","headache","nausea","vomiting","fatigue"},
        "precautions": ["Antimalarial drugs as prescribed","Use mosquito net","Complete full course","Avoid standing water"],
        "description": "A life-threatening disease caused by Plasmodium parasites transmitted by Anopheles mosquitoes.",
    },
    "Gastroenteritis": {
        "triggers": {"diarrhoea","vomiting","nausea","stomach_pain","mild_fever","dehydration"},
        "precautions": ["Hand hygiene","Avoid raw food","ORS hydration","Rest"],
        "description": "Inflammation of the stomach and intestines, typically due to viral or bacterial infection.",
    },
    "Hypertension": {
        "triggers": {"headache","dizziness","blurred_and_distorted_vision","chest_pain","palpitations"},
        "precautions": ["Low sodium diet","Stress management","Regular BP monitoring","Daily exercise"],
        "description": "A condition where blood pressure in the arteries is persistently elevated.",
    },
    "Diabetes Mellitus": {
        "triggers": {"increased_appetite","polyuria","fatigue","weight_loss","blurred_and_distorted_vision","irregular_sugar_level"},
        "precautions": ["Monitor blood sugar","Low GI diet","Regular exercise","Medication compliance"],
        "description": "A chronic disease occurring when the pancreas does not produce enough insulin.",
    },
    "Tuberculosis": {
        "triggers": {"cough","fatigue","weight_loss","sweating","chest_pain","high_fever","blood_in_sputum"},
        "precautions": ["Complete 6-month treatment course","Isolate if active TB","Regular chest X-ray","Nutritious diet"],
        "description": "A serious infection mainly affecting the lungs, caused by Mycobacterium tuberculosis.",
    },
    "Anaemia": {
        "triggers": {"fatigue","pale_skin","dizziness","breathlessness","cold_hands_and_feets","weakness_in_limbs"},
        "precautions": ["Iron-rich diet","Treat underlying cause","Regular CBC check","Vitamin B12 check"],
        "description": "Lack of healthy red blood cells to carry adequate oxygen to body tissues.",
    },
}


def fallback_predict(matched):
    sym_set = set(matched)
    scores = {
        d: round(len(sym_set & info["triggers"]) / max(len(sym_set), len(info["triggers"])) * 100)
        for d, info in FALLBACK.items() if sym_set & info["triggers"]
    }
    if not scores:
        return None, 0, [], "", []
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top, top_conf = ranked[0]
    info = FALLBACK[top]
    others = [{"disease": d, "confidence": c} for d, c in ranked[1:4]]
    return top, top_conf, info["precautions"], info["description"], others


# ─── SCHEMAS ─────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    symptoms: List[str]

    @validator("symptoms")
    def check(cls, v):
        cleaned = [s.strip() for s in v if s.strip()]
        if len(cleaned) < 3:
            raise ValueError("Provide at least 3 symptoms.")
        if len(cleaned) > 5:
            raise ValueError("Provide at most 5 symptoms.")
        return cleaned


class OtherPrediction(BaseModel):
    disease: str
    confidence: int


class PredictResponse(BaseModel):
    disease: str
    confidence: int
    description: str
    precautions: List[str]
    matched_symptoms: List[str]
    other_predictions: List[OtherPrediction]
    model_used: str


# ─── ROUTES ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Disease Prediction API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "symptoms": len(symptom_list)}


@app.get("/symptoms")
def get_symptoms():
    return {
        "symptoms": sorted({s.replace("_", " ") for s in symptom_list if s != "prognosis"}),
        "count": len(symptom_list),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    matched = map_symptoms(req.symptoms)
    if not matched:
        raise HTTPException(422, "No recognisable symptoms found.")

    if model is not None:
        try:
            vec     = build_vector(matched)
            disease = str(model.predict(vec)[0])

            if hasattr(model, "predict_proba"):
                proba      = model.predict_proba(vec)[0]
                classes    = list(model.classes_)
                top_idx    = int(np.argmax(proba))
                confidence = int(round(proba[top_idx] * 100))
                sorted_idx = np.argsort(proba)[::-1]
                others = [
                    OtherPrediction(disease=str(classes[i]), confidence=int(round(proba[i] * 100)))
                    for i in sorted_idx[1:4] if proba[i] > 0.02
                ]
            else:
                confidence, others = 80, []

            description, precautions = "", []
            if description_df is not None:
                row = description_df[description_df.iloc[:, 0].str.lower() == disease.lower()]
                if not row.empty:
                    description = str(row.iloc[0, 1])
            if precaution_df is not None:
                row = precaution_df[precaution_df.iloc[:, 0].str.lower() == disease.lower()]
                if not row.empty:
                    precautions = [str(v) for v in row.iloc[0, 1:].dropna().tolist()]

            return PredictResponse(
                disease=disease, confidence=confidence,
                description=description or f"ML model predicted: {disease}.",
                precautions=precautions or ["Consult a doctor", "Rest well", "Stay hydrated"],
                matched_symptoms=[s.replace("_", " ") for s in matched],
                other_predictions=others, model_used="trained_ml_model",
            )
        except Exception as exc:
            print(f"Model error: {exc} — switching to fallback.")

    disease, confidence, precautions, description, others_raw = fallback_predict(matched)
    if disease is None:
        raise HTTPException(404, "Could not match symptoms to a known disease. Add more specific symptoms.")

    return PredictResponse(
        disease=disease, confidence=confidence, description=description,
        precautions=precautions,
        matched_symptoms=[s.replace("_", " ") for s in matched],
        other_predictions=[OtherPrediction(**o) for o in others_raw],
        model_used="rule_based_fallback",
    )
