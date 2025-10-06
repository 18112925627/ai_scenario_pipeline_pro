import re, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
SUGAR = ['蔗糖','白砂糖','葡萄糖','果糖','乳糖','麦芽糖','葡萄糖浆','果葡糖浆','玉米糖浆','麦芽糊精','蜂蜜','红糖','赤砂糖']
SWEET = ['阿斯巴甜','安赛蜜','三氯蔗糖','甜蜜素','纽甜','糖精钠','赤藓糖醇','木糖醇','山梨糖醇']
TRANS  = ['部分氢化','起酥油','植脂末','氢化植物油']
SALT   = ['食盐','氯化钠','味精','鸡精','小苏打','碳酸氢钠']
PROTEIN_HINT = ['乳清蛋白','分离大豆蛋白','酪蛋白','蛋白粉','浓缩乳清']
CAFFEINE = ['咖啡因','可乐果']
def normalize(text:str)->str:
    if not text: return ''
    t=str(text); t=re.sub(r"\s+","",t); t=t.replace("（","(").replace("）",")"); t=re.sub(r"[，,;；]","、",t); t=re.sub(r"^((配料|成分|原料)[:：])?","",t); return t
def lexicon_features(text:str)->np.ndarray:
    t = normalize(text)
    feats = [
        int(any(k in t for k in SUGAR)),
        int(any(k in t for k in SWEET)),
        int(any(k in t for k in TRANS)),
        int(t.count("食盐")+t.count("氯化钠")+t.count("味精")+t.count("鸡精")>=2),
        int(any(k in t for k in PROTEIN_HINT)),
        int(any(k in t for k in CAFFEINE)),
    ]
    return np.array(feats, dtype=np.float32)
class HybridVectorizer:
    def __init__(self, max_features=5000, ngram_range=(1,2)): self.tfidf=TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    def fit(self, texts): texts=[normalize(x) for x in texts]; self.tfidf.fit(texts); return self
    def transform(self, texts):
        from scipy.sparse import hstack, csr_matrix
        texts=[normalize(x) for x in texts]; X=self.tfidf.transform(texts); L=np.vstack([lexicon_features(x) for x in texts]); return hstack([X, csr_matrix(L)])
