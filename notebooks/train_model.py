import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Creer les dossiers s'ils n'existent pas
os.makedirs("figures", exist_ok=True)
os.makedirs("models", exist_ok=True)

print("--- Etape 2 ---")
df = pd.read_csv("data/patients_dakar.csv")
print(f"Dataset : {df.shape[0]} patients, {df.shape[1]} colonnes")
print(f"\nColonnes : {list(df.columns)}")
print(f"\nDiagnostics :\n{df['diagnostic'].value_counts()}")

le_sexe = LabelEncoder()
le_region = LabelEncoder()
df['sexe_encoded'] = le_sexe.fit_transform(df['sexe'])
df['region_encoded'] = le_region.fit_transform(df['region'])

feature_cols = ['age', 'sexe_encoded', 'temperature', 'tension_sys', 'toux', 'fatigue', 'maux_tete', 'region_encoded']
X = df[feature_cols]
y = df['diagnostic']

print("\n--- Etape 3 ---")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Entrainement : {X_train.shape[0]} patients")
print(f"Test : {X_test.shape[0]} patients")

print("\n--- Etape 4 ---")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Modele entraine !")
print(f"Nombre d'arbres : {model.n_estimators}")
print(f"Nombre de features : {model.n_features_in_}")
print(f"Classes : {list(model.classes_)}")

print("\n--- Etape 5 ---")
y_pred = model.predict(X_test)

comparison = pd.DataFrame({
    'Vrai diagnostic': y_test.values[:10],
    'Prediction': y_pred[:10]
})
print("Comparaison (10 premiers) :\n", comparison)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy : {accuracy:.2%}")

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
print("\nMatrice de confusion :")
print(cm)

print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Prediction du modele')
plt.ylabel('Vrai diagnostic')
plt.title('Matrice de confusion - SenSante')
plt.tight_layout()
plt.savefig('figures/confusion_matrix.png', dpi=150)
plt.close() # on ferme direct pour ne pas bloquer le script
print("Figure sauvegardee dans figures/confusion_matrix.png")

print("\n--- Etape 6 ---")
joblib.dump(model, "models/model.pkl")
joblib.dump(le_sexe, "models/encoder_sexe.pkl")
joblib.dump(le_region, "models/encoder_region.pkl")
joblib.dump(feature_cols, "models/feature_cols.pkl")

size = os.path.getsize("models/model.pkl")
print(f"Modele sauvegarde : models/model.pkl")
print(f"Taille : {size/1024:.1f} Ko")

print("\n--- Etape 7 ---")
model_loaded = joblib.load("models/model.pkl")
le_sexe_loaded = joblib.load("models/encoder_sexe.pkl")
le_region_loaded = joblib.load("models/encoder_region.pkl")
print(f"Modele recharge : {type(model_loaded).__name__}")
print(f"Classes : {list(model_loaded.classes_)}")

nouveau_patient = {
    'age': 28, 'sexe': 'F', 'temperature': 39.5, 'tension_sys': 110,
    'toux': True, 'fatigue': True, 'maux_tete': True, 'region': 'Dakar'
}
sexe_enc = le_sexe_loaded.transform([nouveau_patient['sexe']])[0]
region_enc = le_region_loaded.transform([nouveau_patient['region']])[0]
features = [
    nouveau_patient['age'], sexe_enc, nouveau_patient['temperature'], nouveau_patient['tension_sys'],
    int(nouveau_patient['toux']), int(nouveau_patient['fatigue']), int(nouveau_patient['maux_tete']), region_enc
]

diagnostic = model_loaded.predict([features])[0]
probas = model_loaded.predict_proba([features])[0]
proba_max = probas.max()
print("\n--- Resultat du pre-diagnostic ---")
print(f"Patient : {nouveau_patient['sexe']}, {nouveau_patient['age']} ans")
print(f"Diagnostic : {diagnostic}")
print(f"Probabilite : {proba_max:.1%}")

print("\nProbabilites par classe :")
for classe, proba in zip(model_loaded.classes_, probas):
    bar = '#' * int(proba * 30)
    print(f"{classe:8s} : {proba:.1%} {bar}")

print("\n--- Exercice 1 ---")
importances = model.feature_importances_
for name, imp in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True):
    print(f"{name:20s} : {imp:.3f}")

print("\n--- Exercice 2 ---")
patients = [
    {'age': 20, 'sexe': 'M', 'temperature': 37.0, 'tension_sys': 120, 'toux': False, 'fatigue': False, 'maux_tete': False, 'region': 'Dakar'},
    {'age': 45, 'sexe': 'M', 'temperature': 40.2, 'tension_sys': 130, 'toux': True, 'fatigue': True, 'maux_tete': True, 'region': 'Pikine'},
    {'age': 75, 'sexe': 'F', 'temperature': 37.5, 'tension_sys': 110, 'toux': True, 'fatigue': True, 'maux_tete': False, 'region': 'Rufisque'}
]
for i, p in enumerate(patients):
    s_e = le_sexe_loaded.transform([p['sexe']])[0]
    try:
         r_e = le_region_loaded.transform([p['region']])[0]
    except ValueError:
         r_e = 0
    f = [p['age'], s_e, p['temperature'], p['tension_sys'], int(p['toux']), int(p['fatigue']), int(p['maux_tete']), r_e]
    diag = model_loaded.predict([f])[0]
    print(f"Patient {i+1} : {p} -> Diagnostic: {diag}")
