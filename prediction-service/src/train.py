"""
Entraînement d'un modèle ML (XGBoost) pour prédire le podium.
- Input : data/processed/{train,val,test}.csv + features.txt
- Output : models/model.pkl + rapport de perf
"""

from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)


def load_data():
    features = Path(PROC / "features.txt").read_text().splitlines()
    train = pd.read_csv(PROC / "train.csv")
    val = pd.read_csv(PROC / "val.csv")

    X_train, y_train = train[features], train["podium"]
    X_val, y_val = val[features], val["podium"]
    return X_train, y_train, X_val, y_val, features


def train_model():
    X_train, y_train, X_val, y_val, features = load_data()

    # Modèle XGBoost
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)

    # Prédictions
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    # Scores
    print("Validation set performance:")
    print("Accuracy:", accuracy_score(y_val, y_pred))
    print("F1:", f1_score(y_val, y_pred))
    print("ROC-AUC:", roc_auc_score(y_val, y_proba))
    print("\nRapport détaillé:\n", classification_report(y_val, y_pred))

    # Sauvegarde du modèle
    joblib.dump(model, MODELS / "xgb_podium.pkl")
    print(f"Modèle sauvegardé → {MODELS / 'xgb_podium.pkl'}")


if __name__ == "__main__":
    train_model()
