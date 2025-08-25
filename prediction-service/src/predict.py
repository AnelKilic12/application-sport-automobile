from pathlib import Path
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"

def load_model():
    model = joblib.load(MODELS / "xgb_podium.pkl")
    feats = (PROC / "features.txt").read_text().splitlines()
    return model, feats

def load_all_processed():
    parts = []
    for name in ["train.csv", "val.csv", "test.csv"]:
        p = PROC / name
        if p.exists():
            parts.append(pd.read_csv(p, parse_dates=["date"]))
    if not parts:
        raise FileNotFoundError("No processed datasets found. Run features.py first.")
    return pd.concat(parts, ignore_index=True)

def predict_for_race(season: int, round_: int):
    model, feats = load_model()
    df = load_all_processed()
    race_df = df[(df["season"] == season) & (df["round"] == round_)].copy()
    if race_df.empty:
        return {"season": season, "round": round_, "raceName": None, "predictions": []}

    proba = model.predict_proba(race_df[feats])[:, 1]
    race_df["prob_podium"] = proba
    out = (race_df[["driverId", "constructorId", "raceName", "prob_podium"]]
           .sort_values("prob_podium", ascending=False))
    preds = [
        {"driverId": r.driverId, "constructorId": r.constructorId, "prob_podium": float(r.prob_podium)}
        for r in out.itertuples(index=False)
    ]
    race_name = out.iloc[0]["raceName"] if not out.empty else None
    return {"season": season, "round": round_, "raceName": race_name, "predictions": preds}

def save_predictions_csv(season: int, round_: int, path: Path | None = None) -> Path:
    res = predict_for_race(season, round_)
    rows = [{"season": res["season"], "round": res["round"], "raceName": res["raceName"], **p}
            for p in res["predictions"]]
    df = pd.DataFrame(rows)
    if path is None:
        path = ROOT / f"predictions_{season}_{round_}.csv"
    df.to_csv(path, index=False)
    return path
