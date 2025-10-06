import argparse, pandas as pd, numpy as np, os, json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.preprocessing import MultiLabelBinarizer
from .schema import LABELS

def main(args):
    df = pd.read_csv(args.train_csv); df['labels']=df['labels'].fillna('').apply(lambda s: [x for x in str(s).split('|') if x])
    mlb = MultiLabelBinarizer(classes=LABELS); Y = mlb.fit_transform(df['labels']).astype(float)
    tok = AutoTokenizer.from_pretrained(args.model_name)
    def encode(ex): 
        out = tok(ex['text'], truncation=True, padding='max_length', max_length=256)
        return {**out, "labels": ex['y']}
    d = Dataset.from_dict({"text": df['text'].tolist(), "y": list(Y)}).map(encode, batched=True)
    m = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=len(LABELS), problem_type="multi_label_classification")
    args_tr = TrainingArguments(output_dir=args.out_dir, per_device_train_batch_size=8, learning_rate=2e-5, num_train_epochs=3, logging_steps=50, save_strategy="epoch")
    tr = Trainer(model=m, args=args_tr, train_dataset=d, tokenizer=tok)
    tr.train(); os.makedirs(args.model_dir, exist_ok=True)
    m.save_pretrained(args.model_dir); tok.save_pretrained(args.model_dir)
    with open(os.path.join(args.model_dir,"labels.json"),"w",encoding="utf-8") as f: json.dump(LABELS,f,ensure_ascii=False,indent=2)
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--train_csv", default="data/seed_dataset.csv"); p.add_argument("--model_name", default="hfl/chinese-macbert-base"); p.add_argument("--out_dir", default="runs"); p.add_argument("--model_dir", default="artifacts_bert"); main(p.parse_args())
