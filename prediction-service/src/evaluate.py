from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, classification_report

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"

def main():
    feats = (PROC / "features.txt").read_text().splitlines()
    test = pd.read_csv(PROC / "test.csv")
    X, y = test[feats], test["podium"]

    model = joblib.load(MODELS / "xgb_podium.pkl")
    proba = model.predict_proba(X)[:, 1]
    pred  = (proba >= 0.5).astype(int)

    print("[TEST] ROC-AUC:", roc_auc_score(y, proba))
    print("[TEST] LogLoss:", log_loss(y, proba))   # eps supprimé
    print("[TEST] Brier  :", brier_score_loss(y, proba))
    print("\n[TEST] Rapport:\n", classification_report(y, pred))

if __name__ == "__main__":
    main()
