# api/main.py
# SenSante API - Assistant pré-diagnostic médical
# Lab 5 - FastAPI + ML + Groq LLM

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
import joblib
import numpy as np
import traceback
import os
from dotenv import load_dotenv
from groq import Groq

# ============================================================
# ENV
# ============================================================

load_dotenv()

groq_client = None
groq_api_key = os.getenv("GROQ_API_KEY")

if groq_api_key:
    groq_client = Groq(api_key=groq_api_key)
    print("Client Groq initialise.")
else:
    print("ATTENTION : GROQ_API_KEY non trouvee.")

# ============================================================
# SYSTEM PROMPT (FR + WOLOF)
# ============================================================

SYSTEM_PROMPT = """
Tu es un assistant medical senegalais.

Reponds en melange de francais et de wolof simple,
comme un medecin parlerait a son patient a Dakar.

Maximum 3 phrases.
Sois rassurant.

Ne fais JAMAIS de diagnostic.
Tu expliques uniquement le resultat du modele.
"""

# ============================================================
# SCHEMAS
# ============================================================

class PatientInput(BaseModel):
    age: int = Field(..., ge=0, le=120)
    sexe: str
    temperature: float = Field(..., ge=35, le=42)
    tension_sys: int = Field(..., ge=60, le=250)
    toux: bool
    fatigue: bool
    maux_tete: bool
    frissons: bool
    nausee: bool
    region: str


class ExplainInput(BaseModel):
    diagnostic: str
    probabilite: float
    age: int
    sexe: str
    temperature: float
    region: str


class ExplainOutput(BaseModel):
    explication: str
    modele_llm: str = "llama-3.1-8b-instant"


class DiagnosticOutput(BaseModel):
    diagnostic: str
    probabilite: float
    confiance: str
    message: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "diagnostic": "palu",
                "probabilite": 0.92,
                "confiance": "élevée",
                "message": "Suspicion de paludisme."
            }
        }
    )

# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="SenSante API",
    version="0.5.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# FRONTEND STATIC
# ============================================================

app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/")
def serve_frontend():
    """Servir la page d'accueil."""
    return FileResponse("frontend/index.html")

# ============================================================
# MODEL
# ============================================================

print("Chargement modele...")

model = joblib.load("model.pkl")
le_sexe = joblib.load("encoder_sexe.pkl")
le_region = joblib.load("encoder_region.pkl")
feature_cols = joblib.load("feature_cols.pkl")

# ============================================================
# PREDICT
# ============================================================

@app.post("/predict", response_model=DiagnosticOutput)
def predict(patient: PatientInput):

    sexe_enc = le_sexe.transform([patient.sexe])[0]
    region_enc = le_region.transform([patient.region])[0]

    features = np.array([[
        patient.age,
        sexe_enc,
        patient.temperature,
        patient.tension_sys,
        int(patient.toux),
        int(patient.fatigue),
        int(patient.maux_tete),
        int(patient.frissons),
        int(patient.nausee),
        region_enc
    ]])

    pred = model.predict(features)[0]

    try:
        proba = float(model.predict_proba(features)[0].max())
    except:
        proba = 0.0

    confiance = (
        "élevée" if proba >= 0.8
        else "moyenne" if proba >= 0.5
        else "faible"
    )

    messages = {
        "palu": "Suspicion de paludisme.",
        "grippe": "Suspicion de grippe.",
        "typh": "Suspicion de typhoide.",
        "sain": "Pas de maladie detectee."
    }

    return DiagnosticOutput(
        diagnostic=pred,
        probabilite=round(proba, 2),
        confiance=confiance,
        message=messages.get(pred, "Consultez un medecin.")
    )

# ============================================================
# EXPLAIN (LLM GROQ)
# ============================================================

@app.post("/explain", response_model=ExplainOutput)
def explain(data: ExplainInput):

    if not groq_client:
        return ExplainOutput(
            explication="Service LLM indisponible.",
            modele_llm="aucun"
        )

    user_prompt = f"""
Patient :
- Age : {data.age}
- Sexe : {data.sexe}
- Region : {data.region}
- Temperature : {data.temperature}

Diagnostic :
- {data.diagnostic} ({data.probabilite:.0%})

Explique simplement en wolof + francais.
"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=150
        )

        explication = response.choices[0].message.content

    except Exception as e:
        explication = f"Erreur LLM : {str(e)}"

    return ExplainOutput(explication=explication)

# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    return {"status": "ok"}

# ============================================================
# GLOBAL ERROR HANDLER
# ============================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    traceback.print_exc()

    return JSONResponse(
        status_code=500,
        content={
            "error": "Erreur interne serveur",
            "details": str(exc)
        }
    )
