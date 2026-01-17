# -*- coding: utf-8 -*-
"""
Hard-coded pipeline (clean):
payload -> BERT(CLS) -> (LR) method 추정
      -> [CLS + onehot(method) + Δt] 시퀀스 구성
      -> LSTM 으로 label_attack 분류
Artifacts:
  - cls_train.npy, cls_val.npy
  - Xtr_seq.npy, Ytr_seq.npy, Xvl_seq.npy, Yvl_seq.npy
  - method_report.txt, method_clf.joblib
  - lstm_label_attack.pt, meta.pkl
"""

from __future__ import annotations

import os, warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# =========================
# 전역 설정 (경로/칼럼/HP)
# =========================
TRAIN_CSV = "mnt/data/merged_WAF_noimal_attak_same_del_train.csv"
VAL_CSV   = "mnt/data/merged_WAF_noimal_attak_same_del_val.csv"
ART_DIR   = Path("1109/apt_artifacts"); ART_DIR.mkdir(parents=True, exist_ok=True)

PAYLOAD_COL = "payload"
METHOD_COL  = "method"
TIME_COL    = "event_time"
LABEL_COL   = "label_attack"

SEQ_LEN      = 10
SEQ_STRIDE   = 1

BERT_MODEL   = "bert-base-uncased"
MAX_LEN      = 64
EMB_BATCH    = 64

H_LSTM       = 256
LSTM_LAYERS  = 1
DROPOUT      = 0.1
EPOCHS_LSTM  = 15
BATCH_LSTM   = 128
LR_LSTM      = 1e-3
WEIGHT_DECAY = 1e-4

# (선택) BLAS 쓰레딩 과점유/행 방지
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# =========================
# Torch / HF 로드
# =========================
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device: {DEVICE}", flush=True)


# =========================
# 유틸 함수
# =========================
def load_csv(path: str) -> pd.DataFrame:
    """고정 경로 CSV 로드 (존재 확인 포함)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")
    print(f"[LOAD] {p}", flush=True)
    return pd.read_csv(p)


def parse_time_series(series: pd.Series) -> pd.Series:
    """epoch ms/s 또는 문자열 타임스탬프를 UTC Timestamp로 파싱."""
    def _parse(x):
        if pd.isna(x): return pd.NaT
        try:
            v = float(x)
            if v > 1e12:  # ms
                return pd.to_datetime(int(v), unit="ms", utc=True)
            if v > 1e9:   # s
                return pd.to_datetime(int(v), unit="s", utc=True)
        except Exception:
            pass
        return pd.to_datetime(str(x), errors="coerce", utc=True)
    return series.apply(_parse)


def add_time_features_fixed(df: pd.DataFrame) -> pd.DataFrame:
    """Δt 특징 생성: 정렬→diff(sec)→log1p→표준화→_dt_norm."""
    df = df.copy()
    df["_ts"] = parse_time_series(df[TIME_COL])
    df.sort_values("_ts", inplace=True, kind="mergesort")  # 안정 정렬
    ts = (df["_ts"].astype("int64") // 10**9).astype("float64")
    ts = ts.fillna(ts.median() if not np.isnan(ts.median()) else 0.0)
    dts = ts.diff().fillna(0.0).clip(lower=0.0).astype(np.float32)

    dt_log = np.log1p(dts.values)
    mean, std = float(dt_log.mean()), float(dt_log.std())
    dt_norm = (dt_log - mean) / (std if std > 0 else 1.0)
    df["_dt_norm"] = dt_norm.astype(np.float32)
    return df


def compute_cls_embeddings(texts: List[str],
                           tokenizer: AutoTokenizer,
                           bert: AutoModel) -> np.ndarray:
    """BERT CLS 임베딩 계산."""
    vecs = []
    bert.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), EMB_BATCH), desc="BERT CLS 임베딩"):
            batch = texts[i:i+EMB_BATCH]
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=MAX_LEN, return_tensors="pt")
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            out = bert(**enc)
            cls = out.last_hidden_state[:, 0, :]  # [B, 768]
            vecs.append(cls.detach().cpu().numpy())
    return np.concatenate(vecs, axis=0).astype(np.float32)


def onehot(idx: np.ndarray, K: int) -> np.ndarray:
    oh = np.zeros((len(idx), K), dtype=np.float32)
    oh[np.arange(len(idx)), idx] = 1.0
    return oh


def build_sequences(X: np.ndarray, y: np.ndarray,
                    seq_len: int, stride: int) -> Tuple[np.ndarray, np.ndarray]:
    """슬라이딩 윈도우로 [N,T,D], [N] 구성 (라벨은 윈도우 마지막 시점)."""
    Xs, Ys = [], []
    N = len(X)
    if N < seq_len:
        return np.zeros((0, seq_len, X.shape[1]), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    for start in range(0, N - seq_len + 1, stride):
        end = start + seq_len
        Xs.append(X[start:end])
        Ys.append(y[end - 1])
    return np.stack(Xs), np.array(Ys, dtype=np.int64)


class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).long()
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden: int, num_layers: int,
                 num_classes: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=num_layers, batch_first=True,
                            dropout=(dropout if num_layers > 1 else 0.0))
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x.float())
        _, (hn, _) = self.lstm(z)
        return self.head(hn[-1])


# =========================
# 파이프라인 본문
# =========================
def main() -> None:
    # 1) CSV 로드
    print("[STEP] CSV 로드 중...", flush=True)
    train_df = load_csv(TRAIN_CSV)
    val_df   = load_csv(VAL_CSV)
    print(f"[DONE] Train={len(train_df)}, Val={len(val_df)} rows", flush=True)

    # 필수 칼럼 확인
    req_cols = {PAYLOAD_COL, METHOD_COL, TIME_COL, LABEL_COL}
    for name, df in [("train", train_df), ("val", val_df)]:
        miss = sorted(list(req_cols - set(df.columns)))
        if miss:
            raise ValueError(f"{name} CSV에서 누락된 칼럼: {miss}")

    # 2) 시간 특징
    print("[STEP] 시간 파싱 및 Δt 계산...", flush=True)
    train_df = add_time_features_fixed(train_df)
    val_df   = add_time_features_fixed(val_df)
    print("[DONE] 시간 특징 생성 완료", flush=True)

    # 3) BERT 임베딩
    print(f"[STEP] BERT 모델 로드: {BERT_MODEL}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL, use_fast=True)
    bert = AutoModel.from_pretrained(BERT_MODEL).to(DEVICE)
    print("[DONE] BERT 로드 완료", flush=True)

    print("[STEP] Train BERT 임베딩 계산...", flush=True)
    cls_tr = compute_cls_embeddings(train_df[PAYLOAD_COL].astype(str).tolist(), tokenizer, bert)
    print(f"[DONE] Train 임베딩 shape: {cls_tr.shape}", flush=True)

    print("[STEP] Val BERT 임베딩 계산...", flush=True)
    cls_vl = compute_cls_embeddings(val_df[PAYLOAD_COL].astype(str).tolist(), tokenizer, bert)
    print(f"[DONE] Val 임베딩 shape: {cls_vl.shape}", flush=True)

    np.save(ART_DIR / "cls_train.npy", cls_tr)
    np.save(ART_DIR / "cls_val.npy",   cls_vl)

    # 4) method 분류 (LR만 사용)
    print("[STEP] Method 분류 (LR) 학습...", flush=True)
    from sklearn.preprocessing import LabelEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, classification_report
    import joblib

    le_method = LabelEncoder()
    ytr_m = le_method.fit_transform(train_df[METHOD_COL].astype(str))
    yvl_m = le_method.transform(val_df[METHOD_COL].astype(str))

    lr = LogisticRegression(max_iter=1000)
    lr.fit(cls_tr, ytr_m)

    pred_tr_m = lr.predict(cls_tr)
    pred_vl_m = lr.predict(cls_vl)
    f1_lr = f1_score(yvl_m, pred_vl_m, average="macro")
    best_name, best_clf = "LR", lr

    print(f"[DONE] LR Macro-F1={f1_lr:.4f}", flush=True)

    # 리포트/클래시파이어 저장
    report = []
    report.append("==== Validation (method) ====")
    report.append(f"LR Macro-F1={f1_lr:.6f}\n")
    report.append("-- LR report --")
    report.append(classification_report(yvl_m, pred_vl_m, target_names=le_method.classes_))
    (ART_DIR / "method_report.txt").write_text("\n".join(report), encoding="utf-8")
    print(f"[SAVE] {ART_DIR/'method_report.txt'}", flush=True)

    joblib.dump({"name": best_name, "model": best_clf, "classes_": le_method.classes_},
                ART_DIR / "method_clf.joblib")
    print(f"[SAVE] {ART_DIR/'method_clf.joblib'}", flush=True)

    # 5) LSTM 입력 구성
    print("[STEP] LSTM 입력 구성...", flush=True)
    from sklearn.preprocessing import LabelEncoder as _LabelEncoder

    le_attack = _LabelEncoder()
    ytr_a = le_attack.fit_transform(train_df[LABEL_COL].astype(str))
    yvl_a = le_attack.transform(val_df[LABEL_COL].astype(str))
    C_attack = len(le_attack.classes_)

    K_method = len(le_method.classes_)
    oh_tr = onehot(pred_tr_m, K_method)
    oh_vl = onehot(pred_vl_m, K_method)

    xtr_full = np.concatenate([cls_tr, oh_tr, train_df["_dt_norm"].values.reshape(-1, 1)], axis=1)
    xvl_full = np.concatenate([cls_vl, oh_vl, val_df["_dt_norm"].values.reshape(-1, 1)], axis=1)
    print(f"[INFO] Input dim={xtr_full.shape[1]}, K_method={K_method}, C_attack={C_attack}", flush=True)

    Xtr_seq, Ytr_seq = build_sequences(xtr_full, ytr_a, SEQ_LEN, SEQ_STRIDE)
    Xvl_seq, Yvl_seq = build_sequences(xvl_full, yvl_a, SEQ_LEN, SEQ_STRIDE)
    print(f"[DONE] TrainSeq={Xtr_seq.shape}, ValSeq={Xvl_seq.shape}", flush=True)

    np.save(ART_DIR / "Xtr_seq.npy", Xtr_seq)
    np.save(ART_DIR / "Ytr_seq.npy", Ytr_seq)
    np.save(ART_DIR / "Xvl_seq.npy", Xvl_seq)
    np.save(ART_DIR / "Yvl_seq.npy", Yvl_seq)
    print(f"[SAVE] X/Y seq saved to {ART_DIR}", flush=True)

    # 6) LSTM 학습
    print("[STEP] LSTM 학습 시작...", flush=True)
    train_loader = DataLoader(SeqDataset(Xtr_seq, Ytr_seq), batch_size=BATCH_LSTM, shuffle=True)
    val_loader   = DataLoader(SeqDataset(Xvl_seq, Yvl_seq), batch_size=BATCH_LSTM, shuffle=False)

    model = LSTMClassifier(xtr_full.shape[1], H_LSTM, LSTM_LAYERS, C_attack, DROPOUT).to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=LR_LSTM, weight_decay=WEIGHT_DECAY)
    crit  = nn.CrossEntropyLoss()

    from sklearn.metrics import accuracy_score, f1_score

    def eval_on(loader: DataLoader) -> Tuple[float, float]:
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb).argmax(1)
                ys.append(yb.cpu().numpy()); ps.append(pred.cpu().numpy())
        y = np.concatenate(ys); p = np.concatenate(ps)
        acc = accuracy_score(y, p)
        f1m = f1_score(y, p, average="macro")
        return acc, f1m

    best_f1, best_state = -1.0, None
    for epoch in range(1, EPOCHS_LSTM + 1):
        model.train(); total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optim.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward(); optim.step()
            total += float(loss.item()) * xb.size(0)
        train_loss = total / max(1, len(train_loader.dataset))
        acc, f1m = eval_on(val_loader)
        print(f"[EPOCH {epoch:02d}] loss={train_loss:.4f} | val_acc={{acc:.4f}} | val_f1={{f1m:.4f}}".format(acc=acc, f1m=f1m), flush=True)
        if f1m > best_f1:
            best_f1 = f1m
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    print("[DONE] 학습 완료", flush=True)

    # 7) 저장(모델/메타)
    torch.save(model.state_dict(), ART_DIR / "lstm_label_attack.pt")
    print(f"[SAVE] 모델 저장: {ART_DIR/'lstm_label_attack.pt'}", flush=True)

    meta = {
        "label_attack_classes": np.array(le_attack.classes_).tolist(),
        "method_classes":       np.array(le_method.classes_).tolist(),
        "best_method_model":    "LR",
        "const": {
            "PAYLOAD_COL": PAYLOAD_COL,
            "METHOD_COL":  METHOD_COL,
            "TIME_COL":    TIME_COL,
            "LABEL_COL":   LABEL_COL,
            "SEQ_LEN": SEQ_LEN,
            "SEQ_STRIDE": SEQ_STRIDE,
            "BERT_MODEL": BERT_MODEL,
            "MAX_LEN": MAX_LEN
        }
    }
    import joblib
    joblib.dump(meta, ART_DIR / "meta.pkl")
    print(f"[SAVE] {ART_DIR/'meta.pkl'}", flush=True)

    print("\n=== DONE ===", flush=True)
    print(f"Artifacts -> {ART_DIR}", flush=True)


if __name__ == "__main__":
    main()
