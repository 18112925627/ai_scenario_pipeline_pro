from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from pathlib import Path
import os, json, time, csv, io

from model.schema import LABELS

USE_BERT = os.environ.get("USE_BERT","0") == "1"
if USE_BERT:
    from model.infer_bert import PredictorBERT as Predictor
else:
    from model.infer import Predictor

ART_DIR = Path(os.environ.get("MODEL_DIR","artifacts"))
DATA_DIR = Path("data"); DATA_DIR.mkdir(exist_ok=True)
POOL = DATA_DIR / "pool.jsonl"
FEED = DATA_DIR / "feedback.jsonl"

app = FastAPI(title="FoodPack Scenario AI")
templates = Jinja2Templates(directory="service/templates")

class Nutrition(BaseModel):
    energy_kcal: Optional[float]=None
    fat: Optional[float]=None
    saturatedFat: Optional[float]=None
    carb: Optional[float]=None
    sugars: Optional[float]=None
    protein: Optional[float]=None
    sodium_mg: Optional[float]=None

class PredictIn(BaseModel):
    ingredientsText: str
    nutrition: Optional[Nutrition]=None

class PredictOut(BaseModel):
    scenarios: List[str]
    scores: Dict[str, float]

class FeedbackIn(BaseModel):
    ingredientsText: str
    label: Optional[List[str]] = None
    meta: Optional[Dict[str, Any]] = None

PRED = None
def get_pred():
    global PRED
    if PRED is None:
        PRED = Predictor(model_dir=str(ART_DIR if not USE_BERT else Path("artifacts_bert")))
    return PRED

@app.get("/labels")
def labels(): return {"labels": LABELS}

@app.post("/predict", response_model=PredictOut)
def predict(inp: PredictIn):
    pred = get_pred()
    preds, scores = pred.predict(inp.ingredientsText, threshold=0.5)
    return {"scenarios": preds, "scores": scores}

@app.post("/feedback")
def feedback(inp: FeedbackIn):
    rec = {"t": time.time(), "text": inp.ingredientsText, "label": inp.label, "meta": inp.meta}
    path = FEED if inp.label else POOL
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"ok": True, "stored": str(path)}

@app.get("/al/suggest")
def al_suggest(k:int=50):
    items = []
    if POOL.exists():
        with open(POOL, "r", encoding="utf-8") as f:
            for line in f:
                items.append(json.loads(line))
    return {"items": items[:k]}

@app.get("/export.csv")
def export_csv():
    # Export labeled feedback as CSV: text,labels
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["text","labels"])
    if FEED.exists():
        with open(FEED,"r",encoding="utf-8") as f:
            for line in f:
                r=json.loads(line); 
                if r.get("label"):
                    w.writerow([r.get("text",""), "|".join(r.get("label",[]))])
    buf.seek(0)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv")

@app.get("/label", response_class=HTMLResponse)
def label_ui(request: Request):
    return templates.TemplateResponse("label.html", {"request": request, "labels": LABELS})
