from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.predict import predict_for_race

app = FastAPI(title="F1 Podium Prediction API", version="1.0.0")

class RaceQuery(BaseModel):
    season: int
    round: int

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(q: RaceQuery):
    try:
        return predict_for_race(q.season, q.round)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
