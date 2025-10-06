import os, json
from joblib import load
from .schema import LABELS
class Predictor:
    def __init__(self, model_dir="artifacts"):
        self.vec = load(os.path.join(model_dir, "vectorizer.joblib"))
        self.clf = load(os.path.join(model_dir, "model.joblib"))
        self.labels = LABELS
    def predict(self, text:str, topk=None, threshold=0.5):
        X = self.vec.transform([text])
        try: proba = self.clf.predict_proba(X)[0]
        except Exception:
            import numpy as np
            logits = self.clf.decision_function(X)[0]; proba = 1/(1+np.exp(-logits))
        preds = [self.labels[i] for i,p in enumerate(proba) if p>=threshold]
        scores = { self.labels[i]: float(proba[i]) for i in range(len(self.labels)) }
        if topk: preds = [k for k,_ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topk]]
        return preds, scores
