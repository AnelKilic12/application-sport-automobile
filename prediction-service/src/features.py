"""
Construction du dataset ML pour prédire 'podium' (0/1) à partir des CSV issus de FastF1.
Anti-fuite : on ne regarde que l'historique AVANT chaque course (shift + rolling).
Export : data/processed/{train,val,test}.csv + features.txt
"""

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)


def load_raw():
    races = pd.read_csv(RAW / "races.csv", parse_dates=["date"])
    results = pd.read_csv(RAW / "results.csv", parse_dates=["date"])
    return races, results


def add_podium_label(df: pd.DataFrame) -> pd.DataFrame:
    # 1 si position 1/2/3, sinon 0 (NaN/abandon -> 0)
    df["podium"] = np.where(df["position"].isin([1, 2, 3]), 1, 0)
    return df


def engineer_features():
    races, results = load_raw()

    # Nettoyage de base
    results["position"] = pd.to_numeric(results["position"], errors="coerce")
    results["grid"] = pd.to_numeric(results["grid"], errors="coerce").fillna(0).astype(int)
    results["points"] = pd.to_numeric(results["points"], errors="coerce").fillna(0.0)

    # Jointure robuste
    keep_cols_races = [c for c in ["season", "round", "date", "raceName", "circuitId", "circuitName", "country"] if c in races.columns]
    df = results.merge(
        races[keep_cols_races],
        on=[c for c in ["season", "round", "date"] if c in races.columns and c in results.columns],
        how="left"
    )

    if "raceName" not in df.columns:
        if "circuitName" in df.columns:
            df["raceName"] = df["circuitName"]
        elif "country" in df.columns:
            df["raceName"] = df["country"].fillna("Unknown") + " Grand Prix"
        else:
            df["raceName"] = "Unknown Race"

    # Features simples
    df["dnf"] = np.where(df["position"].isna(), 1, 0)
    df["finish_pos"] = df["position"].fillna(30)

    # ---------- Rolling features ----------
    df = df.sort_values(["driverId", "date"]).reset_index(drop=True)

    def roll_mean(series, win=5):
        return series.shift(1).rolling(win, min_periods=1).mean() # shift(1) pour éviter la fuite

    df["drv_avg_finish_5"] = df.groupby("driverId")["finish_pos"].transform(lambda s: roll_mean(s, 5))
    df["drv_avg_points_5"] = df.groupby("driverId")["points"].transform(lambda s: roll_mean(s, 5))
    df["drv_dnf_rate_5"]   = df.groupby("driverId")["dnf"].transform(lambda s: roll_mean(s, 5))

    # Constructeur
    df = df.sort_values(["constructorId", "date"]).reset_index(drop=True)
    df["con_avg_points_5"] = df.groupby("constructorId")["points"].transform(lambda s: roll_mean(s, 5))

    # Historique pilote-circuit
    df = df.sort_values(["driverId", "circuitId", "date"]).reset_index(drop=True)
    if "circuitId" in df.columns:
        df["drv_circuit_avg_finish"] = df.groupby(["driverId", "circuitId"])["finish_pos"].transform(
            lambda s: s.shift(1).expanding(min_periods=1).mean()
        )
    else:
        df["drv_circuit_avg_finish"] = df.groupby("driverId")["finish_pos"].transform(
            lambda s: s.shift(1).expanding(min_periods=1).mean()
        )

    # Re-trier
    sort_keys = [k for k in ["season", "round", "driverId", "date"] if k in df.columns]
    df = df.sort_values(sort_keys).reset_index(drop=True)

    # Label
    df = add_podium_label(df)

    # Features finales
    feature_cols = [
        "grid",
        "drv_avg_finish_5",
        "drv_avg_points_5",
        "drv_dnf_rate_5",
        "con_avg_points_5",
        "drv_circuit_avg_finish",
    ]

    base_cols = ["season", "round", "date", "raceName", "driverId", "constructorId", "podium"]
    base_cols = [c for c in base_cols if c in df.columns]

    ml = df[base_cols + feature_cols].copy()

    for c in feature_cols:
        ml[c] = pd.to_numeric(ml[c], errors="coerce")
        ml[c] = ml[c].fillna(ml[c].median())

    # Split temporel
    if "season" not in ml.columns:
        raise ValueError("Colonne 'season' absente après préparation; vérifie les CSV d'ingestion.")

    train = ml[ml["season"] <= 2021]
    val   = ml[ml["season"] == 2022]
    test  = ml[ml["season"] >= 2023]

    train.to_csv(PROC / "train.csv", index=False)
    val.to_csv(PROC / "val.csv", index=False)
    test.to_csv(PROC / "test.csv", index=False)
    pd.Series(feature_cols).to_csv(PROC / "features.txt", index=False, header=False)

    print("Datasets exportés → data/processed/{train,val,test}.csv")
    print("Features :", feature_cols)


if __name__ == "__main__":
    engineer_features()
