import os, json, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .schema import LABELS
class PredictorBERT:
    def __init__(self, model_dir="artifacts_bert"):
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.m = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.labels = LABELS
    @torch.no_grad()
    def predict(self, text:str, threshold=0.5):
        enc = self.tok(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        logits = self.m(**enc).logits[0]
        probs = torch.sigmoid(logits).cpu().numpy().tolist()
        preds = [self.labels[i] for i,p in enumerate(probs) if p>=threshold]
        scores = { self.labels[i]: float(probs[i]) for i in range(len(self.labels)) }
        return preds, scores
