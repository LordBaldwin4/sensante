# 🏥 SenSante

> **Assistant intelligent de pré-évaluation médicale adapté au contexte sénégalais**

SenSante est une application web d'aide à la **pré-évaluation de symptômes** développée pour fournir une première indication à partir d'informations simples renseignées par l'utilisateur.

L'application combine un **modèle de Machine Learning**, une **API FastAPI** et un **LLM via Groq** afin de produire une estimation et une explication accessible en français et en wolof.

⚠️ **Important : SenSante est un outil d'aide et ne remplace pas un médecin, un professionnel de santé ou un diagnostic médical.**

---

## 🎯 Objectifs

SenSante a été conçu pour :

* faciliter une première évaluation des symptômes ;
* proposer une estimation basée sur un modèle de Machine Learning ;
* prendre en compte des données adaptées au contexte sénégalais ;
* rendre les résultats compréhensibles grâce à une explication en français et en wolof ;
* fournir une architecture simple pouvant être déployée avec Docker.

---

## ✨ Fonctionnalités

### 🩺 Pré-évaluation

L'utilisateur renseigne notamment :

* âge ;
* sexe ;
* température ;
* tension systolique ;
* toux ;
* fatigue ;
* maux de tête ;
* frissons ;
* nausées ;
* région.

Ces informations sont transformées en caractéristiques numériques puis envoyées au modèle de Machine Learning.

### 🤖 Machine Learning

Le projet utilise un **Random Forest Classifier** avec 100 arbres.

Le modèle est entraîné à partir des données présentes dans :

```text
data/patients_dakar.csv
```

Le script `train.py` encode notamment le sexe et la région avant d'entraîner le modèle. Les modèles et encodeurs sont ensuite sauvegardés avec `joblib`.

Les classes actuellement utilisées comprennent notamment :

* `palu`
* `grippe`
* `typh`
* `sain`

L'API retourne également une probabilité ainsi qu'un niveau de confiance :

* **faible**
* **moyenne**
* **élevée**

---

## 🧠 Explication par IA

Après la prédiction du modèle, SenSante peut utiliser **Groq** avec le modèle :

```text
llama-3.1-8b-instant
```

pour générer une explication simplifiée.

L'objectif est de présenter le résultat dans un langage accessible, avec un mélange de **français et de wolof simple**, tout en évitant de présenter le résultat comme un diagnostic médical.

---

## 🏗️ Architecture

```text
                    ┌─────────────────────┐
                    │      Utilisateur    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Interface Web    │
                    │ frontend/index.html │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      FastAPI        │
                    │     api/main.py     │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
          ┌──────────────────┐   ┌──────────────────┐
          │  Random Forest   │   │   Groq / LLM     │
          │  Machine Learning│   │  Explication     │
          └────────┬─────────┘   └──────────────────┘
                   │
                   ▼
          ┌──────────────────┐
          │ Résultat +       │
          │ probabilité +    │
          │ confiance        │
          └──────────────────┘
```

---

## 📁 Structure du projet

```text
sensante/
│
├── api/
│   └── main.py
│
├── data/
│   └── patients_dakar.csv
│
├── figures/
│
├── frontend/
│   └── index.html
│
├── notebooks/
│
├── sensante/
│
├── train.py
│
├── Dockerfile
├── requirements.txt
├── requierements.txt
└── README.md
```

Le dépôt contient également des notebooks, des figures et des fichiers liés au travail d'analyse et d'expérimentation.

---

## 🛠️ Technologies utilisées

| Technologie             | Utilisation                     |
| ----------------------- | ------------------------------- |
| **Python 3.11**         | Langage principal               |
| **FastAPI**             | API backend                     |
| **Pydantic**            | Validation des données          |
| **scikit-learn**        | Machine Learning                |
| **Random Forest**       | Modèle de classification        |
| **Pandas**              | Manipulation des données        |
| **NumPy**               | Traitement numérique            |
| **Joblib**              | Sauvegarde/chargement du modèle |
| **Groq**                | Service LLM                     |
| **Llama 3.1 8B**        | Génération des explications     |
| **HTML/CSS/JavaScript** | Interface utilisateur           |
| **Docker**              | Conteneurisation                |

Le backend est construit avec FastAPI et expose notamment les routes `/predict`, `/explain` et `/health`.

---

# 🚀 Installation

## 1. Cloner le projet

```bash
git clone https://github.com/LordBaldwin4/sensante.git
cd sensante
```

## 2. Créer un environnement virtuel

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

## 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

# 🔑 Configuration de Groq

Pour utiliser la fonctionnalité d'explication par LLM, définir une clé API Groq.

Créer un fichier `.env` :

```env
GROQ_API_KEY=votre_cle_api
```

L'application charge cette variable d'environnement au démarrage. Si aucune clé n'est disponible, l'API indique que le service LLM est indisponible.

**Ne jamais publier votre clé API dans GitHub.**

---

# 🧠 Entraîner le modèle

Le modèle peut être entraîné avec :

```bash
python train.py
```

Le script charge :

```text
data/patients_dakar.csv
```

puis encode les variables catégorielles et entraîne un `RandomForestClassifier`.

Les fichiers générés sont notamment :

```text
model.pkl
encoder_sexe.pkl
encoder_region.pkl
feature_cols.pkl
```

Le script utilise actuellement les caractéristiques suivantes :

```text
age
sexe
temperature
tension_sys
toux
fatigue
maux_tete
frissons
nausee
region
```

---

# ▶️ Lancer l'application

Une fois les dépendances installées et le modèle généré :

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

L'application sera accessible à :

```text
http://localhost:8000
```

La page d'accueil est servie directement par FastAPI depuis :

```text
frontend/index.html
```

---

# 🐳 Utilisation avec Docker

Le projet fournit également un `Dockerfile` basé sur **Python 3.11 slim** et installe les dépendances depuis `requirements.txt`.

Construire l'image :

```bash
docker build -t sensante .
```

Lancer le conteneur :

```bash
docker run -p 8000:8000 sensante
```

Avec la clé Groq :

```bash
docker run \
  -p 8000:8000 \
  -e GROQ_API_KEY="votre_cle_api" \
  sensante
```

---

# 🔌 API

## `GET /health`

Permet de vérifier que l'API fonctionne.

### Réponse

```json
{
  "status": "ok"
}
```

---

## `POST /predict`

Effectue une prédiction à partir des informations du patient.

### Exemple de requête

```json
{
  "age": 25,
  "sexe": "M",
  "temperature": 38.5,
  "tension_sys": 120,
  "toux": false,
  "fatigue": true,
  "maux_tete": true,
  "frissons": true,
  "nausee": false,
  "region": "Dakar"
}
```

### Exemple de réponse

```json
{
  "diagnostic": "palu",
  "probabilite": 0.92,
  "confiance": "élevée",
  "message": "Suspicion de paludisme."
}
```

La réponse est une **prédiction du modèle**, et non un diagnostic médical. La structure correspond au schéma `DiagnosticOutput` défini dans l'API.

---

## `POST /explain`

Cette route permet de générer une explication du résultat grâce au LLM.

Le système est conçu pour expliquer simplement le résultat en **français + wolof** et pour rappeler que le modèle ne constitue pas un diagnostic.

---

# 📊 Données

Le modèle utilise un jeu de données contenant des informations de patients, notamment des variables démographiques, des symptômes et des mesures physiologiques.

Les données utilisées pour l'entraînement sont chargées depuis :

```text
data/patients_dakar.csv
```

Le choix de variables comme la région vise notamment à prendre en compte le contexte local.

---

# ⚠️ Limites et avertissement médical

SenSante est un **prototype d'aide à la pré-évaluation**.

Il ne doit pas être utilisé pour :

* confirmer une maladie ;
* remplacer une consultation médicale ;
* décider seul d'un traitement ;
* remplacer les services d'urgence ;
* prendre une décision médicale importante.

Les prédictions dépendent directement de la qualité, de la représentativité et des limites du jeu de données utilisé pour entraîner le modèle.

En cas de symptômes importants, persistants ou inquiétants, il est recommandé de consulter un professionnel de santé.

---

# 🔐 Sécurité

Quelques bonnes pratiques sont indispensables avant une utilisation en production :

* ne pas exposer `GROQ_API_KEY` ;
* ne pas publier de données personnelles de patients ;
* utiliser HTTPS ;
* restreindre les origines CORS ;
* ajouter une authentification si nécessaire ;
* journaliser les erreurs sans exposer de données sensibles ;
* valider et anonymiser les données médicales ;
* effectuer une véritable validation clinique du modèle avant tout usage médical réel.

---

# 🧪 Évolution possible

Les prochaines versions pourraient intégrer :

* 📈 une évaluation complète du modèle : précision, rappel, F1-score, matrice de confusion ;
* 🧪 une validation sur un jeu de test indépendant ;
* 🌍 davantage de langues locales ;
* 📱 une interface mobile ;
* 👨‍⚕️ un espace professionnel de santé ;
* 📊 un historique des évaluations ;
* 🔐 une authentification sécurisée ;
* 🗄️ une base de données ;
* 🏥 une meilleure intégration avec les structures de santé ;
* 🤖 l'amélioration et la validation du modèle ML ;
* 🚨 une meilleure détection des situations nécessitant une consultation urgente.

---

# 📚 Documentation API

Lorsque l'application est lancée, FastAPI fournit automatiquement la documentation interactive.

### Swagger UI

```text
http://localhost:8000/docs
```

### ReDoc

```text
http://localhost:8000/redoc
```

---

# 👨‍💻 Développement

Pour contribuer au projet :

```bash
git clone https://github.com/LordBaldwin4/sensante.git
cd sensante

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

Créer ensuite une branche :

```bash
git checkout -b feature/ma-fonctionnalite
```

Effectuer les modifications puis :

```bash
git add .
git commit -m "feat: ajout de ma fonctionnalite"
git push origin feature/ma-fonctionnalite
```

---

# 📄 Licence

Aucune licence open source spécifique n'est actuellement indiquée dans le dépôt.

Avant toute réutilisation ou distribution du projet, il est recommandé d'ajouter une licence explicite au repository.

---

# ❤️ SenSante

SenSante a pour ambition de mettre l'intelligence artificielle au service d'une **première orientation médicale accessible**, en tenant compte du contexte local et linguistique sénégalais.

**L'IA peut aider à orienter. Le professionnel de santé reste indispensable pour diagnostiquer et prendre en charge le patient.**
