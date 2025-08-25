"""
Ingestion F1 via FastF1 (remplace Ergast).
- Télécharge calendrier + résultats Course + Qualif pour une plage d'années.
- Sauvegarde 3 CSV compatibles avec le pipeline :
  data/raw/races.csv
  data/raw/results.csv
  data/raw/qualifying.csv
Docs FastF1 : https://theoehrly.github.io/Fast-F1/
"""

from pathlib import Path
import pandas as pd
import numpy as np
import fastf1
from fastf1.core import Session
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

# Active le cache FastF1 (évite de retélécharger à chaque fois)
CACHE_DIR = ROOT / "data" / "fastf1_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

def _slugify(s: str) -> str:
    return (
        s.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("’", "")
        .replace("'", "")
        .replace(".", "")
        .replace(",", "")
    )

def _load_session_safe(year: int, round_number: int, kind: str) -> Session | None:
    """kind: 'R' (race) ou 'Q' (qualifying)"""
    try:
        ses = fastf1.get_session(year, round_number, kind)
        # Pas besoin de télémétrie; on veut juste les résultats/infos
        ses.load(laps=False, telemetry=False, weather=False, messages=False)
        return ses
    except Exception as e:
        print(f"[WARN] {year} Round {round_number} {kind}: {e}")
        return None

def ingest_year(year: int):
    races_rows = []
    results_rows = []
    quali_rows = []

    # Programme officiel de l'année
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    # On garde uniquement les épreuves avec numéro de manche
    schedule = schedule.dropna(subset=["RoundNumber"])

    for _, ev in schedule.iterrows():
        rnd = int(ev["RoundNumber"])
        race_name = ev["EventName"]
        country = ev.get("Country", "")
        locality = ev.get("Location", "")
        date = pd.to_datetime(ev["EventDate"])

        circuit_id = _slugify(race_name)
        circuit_name = race_name  # FastF1 ne fournit pas toujours un "circuitName" distinct

        races_rows.append({
            "season": int(year),
            "round": rnd,
            "raceName": race_name,
            "circuitId": circuit_id,
            "circuitName": circuit_name,
            "country": country,
            "locality": locality,
            "date": date.date().isoformat(),
            "time": ""  # inconnu ici; OK pour ton pipeline
        })

        # Résultats COURSE
        race = _load_session_safe(year, rnd, "R")
        if race is not None and hasattr(race, "results") and race.results is not None:
            res = race.results  # DataFrame
            # Colonnes typiques: Position, Abbreviation, DriverNumber, TeamName, GridPosition, Status, Points, ...
            for _, r in res.iterrows():
                # position peut être NaN si non classé → on le met à None et positionText = Status
                pos = int(r["Position"]) if pd.notna(r["Position"]) else None
                pos_text = str(r.get("Status", "")) if pd.isna(r["Position"]) else str(int(r["Position"]))
                grid = int(r["GridPosition"]) if pd.notna(r["GridPosition"]) else 0
                points = float(r["Points"]) if pd.notna(r["Points"]) else 0.0

                results_rows.append({
                    "season": int(year),
                    "round": rnd,
                    "raceName": race_name,
                    "date": date.date().isoformat(),
                    # On utilise l'abréviation comme identifiant stable (VER, PER, etc.)
                    "driverId": str(r.get("Abbreviation", "")).lower(),  # ex: "ver"
                    "code": str(r.get("Abbreviation", "")),              # ex: "VER"
                    "givenName": str(r.get("FirstName", "")),
                    "familyName": str(r.get("LastName", "")),
                    "constructorId": _slugify(str(r.get("TeamName", ""))),  # ex: "red_bull_racing"
                    "grid": grid,
                    "position": pos,
                    "positionText": pos_text,  # "1"/"2"/"3" ou "DNF"/"Accident"...
                    "points": points,
                    "status": str(r.get("Status", ""))  # "Finished", "Accident", ...
                })

        # Résultats QUALIF
        qual = _load_session_safe(year, rnd, "Q")
        if qual is not None and hasattr(qual, "results") and qual.results is not None: #hasattr c'est pour éviter les sessions sans résultats et éviter les erreurs dans le cas où la session n'existe pas
            qres = qual.results  # DataFrame
            for _, q in qres.iterrows():
                qpos = int(q["Position"]) if pd.notna(q["Position"]) else None
                quali_rows.append({
                    "season": int(year),
                    "round": rnd,
                    "raceName": race_name,
                    "date": date.date().isoformat(),
                    "driverId": str(q.get("Abbreviation", "")).lower(),
                    "constructorId": _slugify(str(q.get("TeamName", ""))),
                    "q_position": qpos,
                    "Q1": q.get("Q1", None),
                    "Q2": q.get("Q2", None),
                    "Q3": q.get("Q3", None),
                })

    return races_rows, results_rows, quali_rows

def main(start_year=2021, end_year=2025):
    all_races, all_results, all_quali = [], [], []

    for y in tqdm(range(start_year, end_year + 1), desc="Ingestion FastF1"):
        rr, rs, qq = ingest_year(y)
        if rr: all_races.extend(rr)
        if rs: all_results.extend(rs)
        if qq: all_quali.extend(qq)

    if all_races:
        pd.DataFrame(all_races).to_csv(RAW / "races.csv", index=False)
    if all_results:
        pd.DataFrame(all_results).to_csv(RAW / "results.csv", index=False)
    if all_quali:
        pd.DataFrame(all_quali).to_csv(RAW / "qualifying.csv", index=False)

    print("Export terminé → data/raw/{races,results,qualifying}.csv")

if __name__ == "__main__":
    main()
