from pathlib import Path
from predict import predict_for_race, save_predictions_csv

if __name__ == "__main__":
    season, rnd = 2025, 1  # Australia, Melbourne
    res = predict_for_race(season, rnd)
    print(res["raceName"], f"({season} R{rnd})")
    for p in res["predictions"][:5]:
        print(f"{p['driverId']:>10s}  prob_podium={p['prob_podium']:.3f}")
    out = save_predictions_csv(season, rnd)
    print("CSV écrit ->", out)
