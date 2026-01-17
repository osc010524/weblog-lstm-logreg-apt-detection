# -*- coding: utf-8 -*-
"""
Hard-coded evaluation for Logistic Regression (LG) method classifier
- 학습에 사용한 아티팩트(1109/apt_artifacts_same_del)
- 검증 CSV (mnt/data/merged_WAF_noimal_attak_same_del_val.csv)
- 모든 파라미터 하드코딩
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
import matplotlib.pyplot as plt

# ===============================
# [1] 하드코딩된 경로 및 설정
# ===============================
ART_DIR = Path("1109/apt_artifacts")
VAL_CSV = Path("mnt/data/merged_WAF_noimal_attak_val.csv")

# 메타 정보 (BERT, 컬럼명, 등)
BERT_MODEL = "bert-base-uncased"
MAX_LEN = 64
PAYLOAD_COL = "payload"
METHOD_COL = "method"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLS_NPY = ART_DIR / "cls_val.npy"
METHOD_CLF_PATH = ART_DIR / "method_clf.joblib"
META_PATH = ART_DIR / "meta.pkl"

# ===============================
# [2] 유틸
# ===============================
def plot_confusion(cm, labels, out_png: Path, normalize=True, title="Confusion Matrix"):
    if normalize:
        cm = cm.astype(np.float64)
        row_sums = cm.sum(axis=1, keepdims=True) + 1e-12
        cm = cm / row_sums
    plt.figure(figsize=(max(6, len(labels)*0.7), max(5, len(labels)*0.7)))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

# ===============================
# [3] 메타 / 모델 로드
# ===============================
print(f"[LOAD] meta.pkl from {META_PATH}")
meta = joblib.load(META_PATH)
method_classes = meta["method_classes"]
print(f"[INFO] Method classes: {len(method_classes)}")

print(f"[LOAD] method_clf.joblib from {METHOD_CLF_PATH}")
clf_pack = joblib.load(METHOD_CLF_PATH)
clf_model = clf_pack["model"]
clf_name = clf_pack.get("name", "LR")
print(f"[INFO] Classifier={clf_name}")

# ===============================
# [4] 검증 데이터 로드
# ===============================
print(f"[LOAD] Validation CSV from {VAL_CSV}")
df = pd.read_csv(VAL_CSV, low_memory=False)
texts = df[PAYLOAD_COL].astype(str).fillna("").tolist()
y_true_names = df[METHOD_COL].astype(str).fillna("UNK").tolist()

# LabelEncoder 기반 매핑
name2idx = {name: i for i, name in enumerate(method_classes)}
y_true = np.array([name2idx.get(n, -1) for n in y_true_names], dtype=np.int64)
mask_valid = y_true >= 0
if not np.all(mask_valid):
    dropped = int((~mask_valid).sum())
    print(f"[WARN] {dropped} unknown labels dropped.")
    texts = [t for i, t in enumerate(texts) if mask_valid[i]]
    y_true = y_true[mask_valid]

# ===============================
# [5] CLS 임베딩 로드 or 계산
# ===============================
if CLS_NPY.exists():
    print(f"[LOAD] cls_val.npy from {CLS_NPY}")
    X = np.load(CLS_NPY)
    if X.shape[0] != len(texts):
        print("[WARN] Row mismatch, recomputing embeddings.")
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL, use_fast=True)
        bert = AutoModel.from_pretrained(BERT_MODEL).to(DEVICE).eval()
        X = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), 64), desc="BERT CLS 재임베딩"):
                batch = texts[i:i+64]
                enc = tokenizer(batch, padding=True, truncation=True,
                                max_length=MAX_LEN, return_tensors="pt")
                enc = {k: v.to(DEVICE) for k, v in enc.items()}
                out = bert(**enc).last_hidden_state
                cls = out[:, 0, :].detach().cpu().numpy().astype(np.float32)
                X.append(cls)
        X = np.concatenate(X, axis=0)
else:
    print(f"[STEP] Computing CLS embeddings from BERT={BERT_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL, use_fast=True)
    bert = AutoModel.from_pretrained(BERT_MODEL).to(DEVICE).eval()
    X = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), 64), desc="BERT CLS 임베딩"):
            batch = texts[i:i+64]
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=MAX_LEN, return_tensors="pt")
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            out = bert(**enc).last_hidden_state
            cls = out[:, 0, :].detach().cpu().numpy().astype(np.float32)
            X.append(cls)
    X = np.concatenate(X, axis=0)
    np.save(CLS_NPY, X)
    print(f"[SAVE] CLS embeddings -> {CLS_NPY}")

# ===============================
# [6] 예측 및 평가
# ===============================
print("[STEP] Predicting...")
try:
    y_pred = clf_model.predict(X.astype(np.float32))
except Exception:
    y_pred = clf_model.predict(X.astype(np.float64))

acc = accuracy_score(y_true, y_pred)
f1m = f1_score(y_true, y_pred, average="macro")
print("\n=== LG Method Classification ===")
print(f"Accuracy : {acc:.6f}")
print(f"Macro-F1 : {f1m:.6f}\n")

rep = classification_report(y_true, y_pred, target_names=method_classes, digits=4, zero_division=0)
print(rep)

# ===============================
# [7] 혼동행렬 시각화 + 오분류 저장
# ===============================
cm = confusion_matrix(y_true, y_pred, labels=list(range(len(method_classes))))
out_png = ART_DIR / "method_confusion_matrix.png"
plot_confusion(cm, method_classes, out_png, normalize=True, title="LG Method Confusion (Normalized)")
print(f"[SAVE] Confusion matrix -> {out_png}")

wrong = np.where(y_true != y_pred)[0]
if len(wrong) > 0:
    wrong_df = pd.DataFrame({
        "payload": np.array(texts)[wrong],
        "true_method": [method_classes[i] for i in y_true[wrong]],
        "pred_method": [method_classes[i] for i in y_pred[wrong]],
    })
    out_wrong = ART_DIR / "method_misclassified_samples.csv"
    wrong_df.to_csv(out_wrong, index=False, encoding="utf-8-sig")
    print(f"[SAVE] Misclassified samples -> {out_wrong} (n={len(wrong)})")
else:
    print("[INFO] No misclassified samples.")

# ===============================
# [8] 텍스트 리포트 저장
# ===============================
out_txt = ART_DIR / "method_eval_report.txt"
with out_txt.open("w", encoding="utf-8") as f:
    f.write("==== Method Classification (LG) ====\n")
    f.write(f"Accuracy   : {acc:.6f}\n")
    f.write(f"Macro-F1   : {f1m:.6f}\n\n")
    f.write(rep)
print(f"[SAVE] Report -> {out_txt}")

print("\n=== DONE ===")
