from pathlib import Path
import pandas as pd
import joblib
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
OUT = ROOT / "reports"
OUT.mkdir(parents=True, exist_ok=True)

def main(use_split="val"):  # "val" ou "test"
    # 1) Charger features et jeu choisi
    feats = (PROC / "features.txt").read_text().splitlines()
    df = pd.read_csv(PROC / f"{use_split}.csv")
    X, y = df[feats], df["podium"]

    # 2) Charger le modèle
    model = joblib.load(MODELS / "xgb_podium.pkl")

    # 3) Importance par permutation (plus robuste que l'importance "gain" par défaut)
    r = permutation_importance(
        model, X, y,
        n_repeats=20, random_state=42, scoring="roc_auc"  # on mesure la chute d'AUC
    )

    imp = (
        pd.DataFrame({"feature": feats, "importance_mean": r.importances_mean, "importance_std": r.importances_std})
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    imp.to_csv(OUT / f"feature_importance_{use_split}.csv", index=False)

    # 4) Graphique (barres)
    top = imp.head(10)
    plt.figure(figsize=(8,4.5))
    plt.barh(top["feature"][::-1], top["importance_mean"][::-1])
    plt.xlabel("Permutation importance (Δ AUC)")
    plt.title(f"Top-10 features – {use_split} set")
    plt.tight_layout()
    plt.savefig(OUT / f"feature_importance_{use_split}.png", dpi=150)
    print(f"Export → {OUT / f'feature_importance_{use_split}.csv'} / {OUT / f'feature_importance_{use_split}.png'}")

if __name__ == "__main__":
    main("val")   # ou main("test")
