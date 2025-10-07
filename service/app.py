from fastapi import FastAPI, Request, HTTPException           # ⬅️ 新增 HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from pathlib import Path
import os, json, time, csv, io

# ⬇️ 新增依赖
import requests
from fastapi.middleware.cors import CORSMiddleware

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

# ⬇️ 允许小程序直连（可按需把 "*" 换成你的前端域名）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# ⬇️ OCR 请求模型（前端只传“纯 base64”，不要带 data:image/... 前缀）
class OCRReq(BaseModel):
    imageBase64: str

PRED = None
def get_pred():
    global PRED
    if PRED is None:
        PRED = Predictor(model_dir=str(ART_DIR if not USE_BERT else Path("artifacts_bert")))
    return PRED

# --- 基础与业务接口 ---
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

# --- 新增：根路径，避免 / 404，可自检可见 ---
@app.get("/")
def root():
    return {"status": "ok", "endpoints": ["/labels", "/predict", "/ocr", "/feedback", "/al/suggest", "/export.csv", "/label"]}

# --- 新增：OCR 路由（后端代调用第三方 OCR，如 OCR.Space） ---
@app.post("/ocr")
def ocr(req: OCRReq):
    api_key = os.environ.get("OCR_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OCR_API_KEY not set")

    # 容错：如果前端带了 data:image/... 前缀，这里剥掉
    b64 = (req.imageBase64 or "").strip()
    if not b64:
        raise HTTPException(status_code=400, detail="empty image")
    if b64.startswith("data:"):
        try:
            b64 = b64.split(",", 1)[1]
        except Exception:
            raise HTTPException(status_code=400, detail="invalid base64 data url")

    data_url = "data:image/jpeg;base64," + b64
    payload = {
        "language": "chs",          # 中文优先；中英混排也可用 chs
        "isOverlayRequired": "false",
        "base64Image": data_url,
        "scale": "true"
    }
    try:
        r = requests.post(
            "https://api.ocr.space/parse/image",
            data=payload,
            headers={"apikey": api_key},
            timeout=20
        )
        r.raise_for_status()
        j = r.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"OCR upstream error: {e}")

    if j.get("IsErroredOnProcessing"):
        msg = j.get("ErrorMessage") or j.get("ErrorDetails") or "OCR failed"
        raise HTTPException(status_code=502, detail=f"OCR upstream: {msg}")

    texts = []
    for item in j.get("ParsedResults") or []:
        t = (item or {}).get("ParsedText") or ""
        if t:
            texts.append(t)

    return {"text": "\n".join(texts).strip()}
