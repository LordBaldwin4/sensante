# api/main.py
# SenSante API - Assistant pré-diagnostic médical
# Lab 3 - Intégration de Modèles IA - ESP/UCAD

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
import joblib
import numpy as np
import traceback
import os
from dotenv import load_dotenv
from groq import Groq

# Charger les variables d'environnement
load_dotenv()

# Client Groq (charge au demarrage)
groq_client = None
groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key:
    groq_client = Groq(api_key=groq_api_key)
    print("Client Groq initialise.")
else:
    print("ATTENTION : GROQ_API_KEY non trouvee. /explain sera desactive.")


# --- Schemas Pydantic ---
class PatientInput(BaseModel):
    age: int = Field(..., ge=0, le=120)
    sexe: str = Field(...)
    temperature: float = Field(..., ge=35.0, le=42.0)
    tension_sys: int = Field(..., ge=60, le=250)
    toux: bool = Field(...)
    fatigue: bool = Field(...)
    maux_tete: bool = Field(...)
    frissons: bool = Field(...)
    nausee: bool = Field(...)
    region: str = Field(...)


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
                "message": "Suspicion de paludisme. Consultez rapidement."
            }
        }
    )


class ExplainInput(BaseModel):
    diagnostic: str = Field(..., description="Diagnostic predit par le modele")
    probabilite: float = Field(..., description="Probabilite du diagnostic")
    age: int = Field(...)
    sexe: str = Field(...)
    temperature: float = Field(...)
    region: str = Field(...)


class ExplainOutput(BaseModel):
    explication: str = Field(..., description="Explication en francais")
    modele_llm: str = Field(default="llama-3.1-8b-instant", description="Modele LLM utilise")


# --- Application FastAPI ---
app = FastAPI(
    title="SenSante API",
    description="Assistant pré-diagnostic médical pour le Sénégal",
    version="0.4.0"
)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Handler d'erreurs global ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)}
    )

# --- Chargement du modèle ---
print("Chargement du modele...")
try:
    model = joblib.load("models/model.pkl")
    le_sexe = joblib.load("models/encoder_sexe.pkl")
    le_region = joblib.load("models/encoder_region.pkl")
    feature_cols = joblib.load("models/feature_cols.pkl")
    print(f"Modele charge    : {list(model.classes_)}")
    print(f"Sexe classes     : {list(le_sexe.classes_)}")
    print(f"Region classes   : {list(le_region.classes_)}")
except Exception as e:
    print(f"ERREUR chargement modele : {e}")
    raise


# --- System Prompt ---
SYSTEM_PROMPT = """Tu es un assistant medical senegalais.
Tu recois un diagnostic et des donnees patient.
Explique le resultat en francais simple,
comme un medecin parlerait a son patient.
Sois rassurant mais recommande toujours une consultation medicale.
Maximum 3 phrases.
Ne fais JAMAIS de diagnostic toi-meme.
Tu expliques uniquement le diagnostic fourni."""


# --- Routes ---

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "SenSante API is running"}


@app.get("/model-info")
def model_info():
    return {
        "type_modele": type(model).__name__,
        "nombre_arbres": model.n_estimators,
        "classes_possibles": list(model.classes_),
        "nombre_features": len(feature_cols)
    }


@app.post("/predict", response_model=DiagnosticOutput)
def predict(patient: PatientInput):

    # Encoder le sexe
    try:
        sexe_enc = le_sexe.transform([patient.sexe])[0]
    except ValueError:
        return DiagnosticOutput(
            diagnostic="erreur",
            probabilite=0.0,
            confiance="aucune",
            message=f"Sexe invalide : '{patient.sexe}'. Valeurs acceptees : {list(le_sexe.classes_)}"
        )

    # Encoder la région
    try:
        region_enc = le_region.transform([patient.region])[0]
    except ValueError:
        return DiagnosticOutput(
            diagnostic="erreur",
            probabilite=0.0,
            confiance="aucune",
            message=f"Region inconnue : '{patient.region}'. Valeurs acceptees : {list(le_region.classes_)}"
        )

    # Construire le vecteur de features
    features = np.array([[
        patient.age, sexe_enc, patient.temperature,
        patient.tension_sys, int(patient.toux),
        int(patient.fatigue), int(patient.maux_tete),
        int(patient.frissons), int(patient.nausee),
        region_enc
    ]])

    # Prédiction
    diagnostic = model.predict(features)[0]

    try:
        proba_max = float(model.predict_proba(features)[0].max())
    except Exception:
        proba_max = 0.0

    # Niveau de confiance
    confiance = ("élevée" if proba_max >= 0.8
                 else "moyenne" if proba_max >= 0.5
                 else "faible")

    # Messages de recommandation
    messages = {
        "palu":   "Suspicion de paludisme. Consultez rapidement.",
        "grippe": "Suspicion de grippe. Repos et hydratation.",
        "typh":   "Suspicion de typhoide. Consultation necessaire.",
        "sain":   "Pas de pathologie detectee."
    }

    return DiagnosticOutput(
        diagnostic=diagnostic,
        probabilite=round(proba_max, 2),
        confiance=confiance,
        message=messages.get(diagnostic, "Consultez un professionnel de sante.")
    )


@app.post("/explain", response_model=ExplainOutput)
def explain(data: ExplainInput):
    """Expliquer un diagnostic en francais avec un LLM."""
    if not groq_client:
        return ExplainOutput(
            explication="Service d'explication indisponible. Cle API non configuree.",
            modele_llm="aucun"
        )

    user_prompt = (
        f"Patient : {data.sexe}, {data.age} ans, region {data.region}\n"
        f"Temperature : {data.temperature}C\n"
        f"Diagnostic du modele : {data.diagnostic} (probabilite {data.probabilite:.0%})\n"
        f"Explique ce resultat au patient."
    )

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=200,
            temperature=0.3
        )
        explication = response.choices[0].message.content
    except Exception as e:
        explication = f"Erreur lors de l'appel au LLM : {str(e)}"

    return ExplainOutput(explication=explication)