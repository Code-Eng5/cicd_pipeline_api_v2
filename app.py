from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os

# Load artifacts
ARTIFACTS_DIR = "deployment_artifacts"
preprocessor = joblib.load(os.path.join(ARTIFACTS_DIR, "preprocess.pkl"))
model = joblib.load(os.path.join(ARTIFACTS_DIR, "lgbm.pkl"))

job_freq_map = joblib.load(os.path.join(ARTIFACTS_DIR, "job_freq_map.pkl"))
stage_freq_map = joblib.load(os.path.join(ARTIFACTS_DIR, "stage_freq_map.pkl"))
branch_freq_map = joblib.load(os.path.join(ARTIFACTS_DIR, "branch_freq_map.pkl"))

app = FastAPI(title="CI/CD Pipeline Prediction API")

class PipelineInput(BaseModel):
    job_name: str
    stage_name: str
    branch: str
    environment: str
    user: str

@app.get("/")
def health():
    return {"status": "running"}

@app.post("/predict")
def predict(data: PipelineInput):
    try:
        df = pd.DataFrame([data.dict()])
        df["job_freq"] = df["job_name"].map(job_freq_map).fillna(1)
        df["stage_freq"] = df["stage_name"].map(stage_freq_map).fillna(1)
        df["branch_freq"] = df["branch"].map(branch_freq_map).fillna(1)

        X = preprocessor.transform(df)
        prob = model.predict_proba(X)[0][1]

        THRESHOLD = 0.4
        return {
            "prediction": "Failure" if prob >= THRESHOLD else "Success",
            "confidence": round(float(prob), 3)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
