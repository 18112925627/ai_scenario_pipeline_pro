import argparse, json, os
import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from .feature import HybridVectorizer
from .schema import LABELS
def load_df(path):
    df = pd.read_csv(path)
    df['labels'] = df['labels'].fillna('').apply(lambda s: [x for x in str(s).split('|') if x])
    return df
def main(args):
    df = load_df(args.train_csv)
    mlb = MultiLabelBinarizer(classes=LABELS)
    Y = mlb.fit_transform(df['labels'])
    vec = HybridVectorizer(max_features=6000, ngram_range=(1,2)).fit(df['text'].tolist())
    X = vec.transform(df['text'].tolist())
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=(Y.sum(axis=1)>0))
    clf = OneVsRestClassifier(LogisticRegression(max_iter=200))
    clf.fit(Xtr, Ytr); Yhat = clf.predict(Xte)
    print(classification_report(Yte, Yhat, target_names=LABELS, zero_division=0))
    print("micro-F1:", f1_score(Yte, Yhat, average='micro', zero_division=0))
    print("macro-F1:", f1_score(Yte, Yhat, average='macro', zero_division=0))
    os.makedirs(args.model_dir, exist_ok=True)
    dump(vec, os.path.join(args.model_dir, "vectorizer.joblib"))
    dump(clf, os.path.join(args.model_dir, "model.joblib"))
    with open(os.path.join(args.model_dir, "labels.json"), "w", encoding="utf-8") as f: json.dump(LABELS, f, ensure_ascii=False, indent=2)
if __name__ == "__main__":
    import argparse; p=argparse.ArgumentParser(); p.add_argument("--train_csv", default="data/seed_dataset.csv"); p.add_argument("--model_dir", default="artifacts"); main(p.parse_args())
