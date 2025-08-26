# F1 Podium Prediction Service

Micro-service de prédiction des podiums en Formule 1, basé sur **FastF1**, **pandas**, **XGBoost** et **FastAPI**.  
Ce projet a été développé dans le cadre d’un travail de Bachelor (HEG Genève, 2025).

---

# RacePro – Modèle IA de prédiction de podium (XGBoost)

Ce projet entraîne un modèle XGBoost pour estimer la probabilité de podium pour chaque pilote d’un Grand Prix de F1.
Il inclut : pipelines d’ingestion (FastF1), features, entraînement/évaluation, et un service FastAPI exposant l’endpoint /predict.

# Fonctionnalités

- Ingestion des données historiques (FastF1) et export en CSV.

- Construction de features (ex. grid, moyennes mobiles 5 courses, taux de DNF, historique pilote-circuit, forme écurie).

- Entraînement et sauvegarde du modèle (xgb_podium.pkl).

- Évaluation complète (Accuracy, F1, ROC-AUC, LogLoss, Brier score).

- API REST FastAPI pour récupérer les probabilités de podium d’une course à venir.

- Démo terminal et export CSV des prédictions par course (predictions_<saison>_<manche>.csv).

# Prérequis

- Python 3.10+
- pip ou uv/pipenv/poetry

## Installation 

    git clone <URL_DU_DEPOT>
    cd prediction-service
    python -m venv .venv
    source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
    pip install -r requirements.txt

### requirements.txt (indicatif) :
    
    pandas
    numpy
    scikit-learn
    xgboost
    fastf1
    fastapi
    uvicorn
    joblib
    matplotlib

# Arborescence du projet

    prediction-service/
    ├── api/
    │ ├── pycache/
    │ └── app.py # API FastAPI exposant l'endpoint /predict
    ├── data/
    │ ├── fastf1_cache/ # Cache FastF1 par saison
    │ │ ├── 2021/
    │ │ ├── 2022/
    │ │ ├── 2023/
    │ │ ├── 2024/
    │ │ └── 2025/
    │ ├── fastf1_http_cache.sqlite 
    │ ├── interim/ # Données temporaires
    │ ├── processed/ # Données transformées pour l'entraînement
    │ │ ├── features.txt
    │ │ ├── test.csv
    │ │ ├── train.csv
    │ │ └── val.csv
    │ └── raw/ # Données brutes (ingestion)
    │ ├── qualifying.csv
    │ ├── races.csv
    │ └── results.csv
    ├── models/
    │ └── xgb_podium.pkl # Modèle XGBoost entraîné
    ├── notebooks/ # Expérimentation (optionnel)
    ├── reports/
    │ ├── feature_importance_val.csv
    │ └── feature_importance_val.png
    ├── src/
    │ ├── pycache/
    │ ├── evaluate.py # (4) Évaluation du modèle sur le jeu de test
    │ ├── features.py (2) Construction des features (grid, moyennes, DNF…)
    │ ├── importance.py (5) Analyse de l’importance des features (permutation, SHAP…)
    │ ├── ingest.py # (1) Ingestion des données avec FastF1
    │ ├── predict_demo.py (6) Démo CLI : prédictions pour une course donnée
    │ ├── predict.py (7) Fonctions de prédiction (utilisées par l’API)
    │ └── train.py (3) Entraînement du modèle XGBoost
    ├── predictions_2023_6.csv # Exemple de sortie prédiction (Monaco 2023)
    ├── predictions_2025_1.csv # Exemple de sortie prédiction (Australie 2025)
    ├── README.md
    └── requirements.txt

# Flux de bout en bout 
## 1. Ingestion (FastF1 -> CSV)
    
    python src/ingest.py
    
    (Exporte dans data/raw/ des CSV du type : races.csv, results.csv, qualifying.csv.)

    Note : l’API Ergast n’étant plus disponible, l’ingestion repose sur FastF1.

## 2. Features
    
    python src/features.py
    
    Construit les variables clés, par ex :
    - grid (position de départ)

    - drv_avg_points_5, drv_avg_finish_5 (forme récente pilote – rolling 5)

    - con_avg_points_5 (forme récente écurie – rolling 5)

    - drv_dnf_rate_5 (taux d’abandon récent – Did Not Finish)

    - drv_circuit_avg_finish (historique pilote–circuit)

    Sorties dans /data/processed

## 3. Entraînement
    
    python src/train.py

    Sorties affichées (validation) :

        Accuracy ≈ 0.88

        F1 ≈ 0.58

        ROC-AUC ≈ 0.90

    Le modèle est sauvegardé sous models/xgb_podium.pkl.
    Un graphique d’importance par permutation est exporté dans figures/.

## 4. Évaluation (jeu de test)

    python src/evaluate.py

    Exemple de résultats (test) :

        Accuracy ≈ 0.94

        F1 (classe podium) ≈ 0.53

        ROC-AUC ≈ 0.90

        LogLoss ≈ 0.17

        Brier score ≈ 0.049

    Un rapport classification_report par classe est affiché en console.

## 5. Importance des features

    python src/importance.py

    Analyse l’importance des variables  
    Export des résultats dans /reports :
    - feature_importance_val.csv  
    - feature_importance_val.png

## 5. Démo CLI (ex Monaco 2023 R6)

    python src/predict_demo.py

    Affiche dans le terminal les Top-N pilotes avec prob_podium, et exporte data/predictions/predictions_2023_6.csv.

# API - FastAPI

    uvicorn src.api:app --reload

    Swagger UI : http://127.0.0.1:8000/docs

## Endpoint
    
    POST /predict – retourne la probabilité de podium par pilote pour une course.

### Request (JSON)

    {
        "season": 2023,
        "round": 6
    }

### Response (extrait)

    {
        "season": 2023,
        "round": 6,
        "raceName": "Monaco Grand Prix",
        "predictions": [
            { "driverId": "ver", "constructorId": "red_bull_racing", "prob_podium": 0.6685 },
            { "driverId": "alo", "constructorId": "aston_martin",   "prob_podium": 0.6444 },
            { "driverId": "oco", "constructorId": "alpine",          "prob_podium": 0.4219 }
        ]
    }

### cUrl

    curl -X 'POST' \
        'http://127.0.0.1:8000/predict' \
        -H 'accept: application/json' \
        -H 'Content-Type: application/json' \
        -d '{
        "season": 2023,
        "round": 6
    }'

# Métriques et interprétation

    - ROC-AUC (~0.90) : bonne séparation podium vs non-podium.

    - LogLoss (~0.17) & Brier (~0.049) : probabilités bien calibrées.

    - F1 (podium) (~0.53) : la classe minoritaire reste difficile (déséquilibre).

    L’importance par permutation montre grid comme variable la plus déterminante, suivie de la forme récente pilote/écurie et de l’historique pilote-circuit.

# Déséquilibre de classes

    Les podiums étant rares, le jeu est déséquilibré. Le code applique :

        - pondération de classe / réglages XGBoost,

        - features ciblant la forme récente,

        - évaluation multi-métriques (pas d’accuracy seule).

# Reproductibilité

    - Fixer les graines aléatoires (numpy, sklearn, xgboost).

    - Versionner : code, modèle, features, données utilisées pour l’entraînement.

# Dépannage

    - Pas de course trouvée : vérifier season/round (et disponibilité FastF1).

    - Modèle introuvable : lancer python src/train.py pour générer models/xgb_podium.pkl.

    - Données manquantes : relancer l’ingestion puis features.py.