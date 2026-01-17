#!/usr/bin/env python3
# auth_collect_and_classify_server.py (final)
# -*- coding: utf-8 -*-
"""
두 가지 모드 지원:
1) 서버 모드(기본): TCP 인증 수신 → 시퀀스(최근 N건) 기반 예측 → detections.csv 기록
2) 파일 모드: 과거 Apache 액세스 로그 파일을 읽어 동일 파이프라인으로 일괄 예측 → detections.csv 기록

실행 예)
  # 서버 모드
  python auth_collect_and_classify_server.py

  # 파일 모드 (기존 로그 파일 분석)
  python auth_collect_and_classify_server.py --from-file ./old_access.log

옵션)
  --from-file FILE         : FILE을 읽어 일괄 분석
  --encoding ENC           : 파일 인코딩 (기본 utf-8, 실패 시 latin1 자동 폴백)
  --no-raw                 : 원문 로그(collected_access.log) 저장하지 않음
  --dry-run                : CSV에 쓰지 않고 콘솔에만 출력
"""

# ===== 환경 변수 (토크나이저 관련 이슈 회피) =====
import os
os.environ["TRANSFORMERS_USE_FAST"] = "0"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# ===== 기본 모듈 =====
import argparse
import socketserver
import threading
import json
import datetime
import re
import csv
import glob
from pathlib import Path
from typing import Optional, Dict, Tuple

# ===== ML/Deps =====
import numpy as np
import joblib
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# =========================
# 하드코딩 파라미터
# =========================
# --- 수신 서버 설정 ---
LISTEN_HOST   = "192.168.0.91"
LISTEN_PORT   = 10514
EXPECTED_USER = "user1"
EXPECTED_PASS = "user1234"   # 운영 환경에서는 TLS/토큰 등으로 교체 권장

# --- 파일 경로 ---
OUT_RAW_LOG   = Path("./collected_access.log")   # 원문 로그 저장
OUT_CSV       = Path("./detections.csv")         # 예측 결과 저장

# --- 모델/아티팩트 ---
CACHE_DIR     = Path("./kmens_method_label_attack_cache_2")
MODEL_NAME    = "microsoft/deberta-v3-base"
MAX_LEN       = 64
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 대표 탐지명 매핑 CSV (있으면 최우선 사용)
REP_METHOD_CSV = Path("./kmens_method_label_attack_cache_2/kmeans_eval_microsoft_deberta-v3-base_L64/cluster_to_rep_method.csv")

# --- CSV 헤더 ---
CSV_HEADER = [
    "time", "client_ip", "request", "status", "bytes",
    "pred_cluster", "pred_method", "pred_label", "prob_attack", "prob_normal"
]

# --- 시퀀스 설정 (학습과 동일하게) ---
SEQ_LEN    = 10
SEQ_STRIDE = 1   # 현재는 1로 고정

# =========================
# Apache 로그 파서 & 페이로드 추출
# =========================
APACHE_RE = re.compile(
    r'(?P<ip>\S+)\s+\S+\s+\S+\s+\[(?P<time>[^\]]+)\]\s+'
    r'"(?P<request>[^"]*)"\s+(?P<status>\d{3})\s+(?P<size>\S+)'
    r'(?:\s+"(?P<referer>[^"]*)"\s+"(?P<agent>[^"]*)")?'
)

def parse_apache_line(line: str) -> Optional[Dict[str, str]]:
    m = APACHE_RE.match(line)
    if not m:
        return None
    return m.groupdict()

def parse_apache_time_to_epoch(ts_str: Optional[str]) -> float:
    """
    Apache 공통 로그 형식 예: 10/Oct/2000:13:55:36 -0700
    가능하면 타임존까지 반영, 실패 시 0.0
    """
    if not ts_str:
        return 0.0
    try:
        # "10/Oct/2000:13:55:36 -0700"
        # 파이썬 %z는 ±HHMM 지원
        dt = datetime.datetime.strptime(ts_str, "%d/%b/%Y:%H:%M:%S %z")
        return dt.timestamp()
    except Exception:
        try:
            # 타임존 없는 케이스 보정: 첫 토큰만 파싱
            base = ts_str.split()[0]
            dt = datetime.datetime.strptime(base, "%d/%b/%Y:%H:%M:%S")
            # naive → UTC로 가정
            return dt.replace(tzinfo=datetime.timezone.utc).timestamp()
        except Exception:
            return 0.0

def extract_payload_from_request(req: Optional[str], fallback_line: str) -> str:
    """기본: 'METHOD PATH' 형태. 실패 시 따옴표 구간/전체 라인에서 대체 추출."""
    if req:
        parts = req.split(" ")
        if len(parts) >= 2:
            return f"{parts[0]} {parts[1]}"
        return req.strip()
    qm = re.findall(r'"([^"]+)"', fallback_line)
    for seg in qm:
        if seg and seg[0:4].upper() in ("GET ", "POST", "PUT ", "DELE", "HEAD", "OPTI", "PATC"):
            parts = seg.split(" ")
            if len(parts) >= 2:
                return f"{parts[0]} {parts[1]}"
            return seg.strip()
    return fallback_line[:256]

# =========================
# LSTM (per-line token 방식) - 기존
# =========================
class LSTMClassifier(nn.Module):
    def __init__(self, input_h, lstm_hidden=256, lstm_layers=1, cluster_k=10,
                 cluster_emb=64, num_classes=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_h, hidden_size=lstm_hidden,
                            num_layers=lstm_layers, batch_first=True, bidirectional=False)
        self.cluster_emb = nn.Embedding(cluster_k, cluster_emb)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden + cluster_emb + 1, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    def forward(self, tokens, masks, clusters, tfeat):
        lengths = masks.sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(tokens, lengths, batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        last_h = h_n[-1]
        cemb = self.cluster_emb(clusters)
        x = torch.cat([last_h, cemb, tfeat.unsqueeze(1)], dim=1)
        return self.fc(x)

# =========================
# LSTM (시퀀스 D-입력 방식) - 학습과 동일
# =========================
class LSTMSeqClassifier(nn.Module):
    def __init__(self, input_dim, hidden=256, num_layers=1, num_classes=2, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=num_layers, batch_first=True,
                            dropout=(dropout if num_layers > 1 else 0.0))
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes)
        )
    def forward(self, x):                # x: [B, T, D]
        z, _ = self.lstm(self.proj(x))  # z: [B, T, H]
        h = z[:, -1, :]                 # 마지막 시점
        return self.head(h)

# =========================
# Artefact 로드 (공용)
# =========================
def latest_file(pattern: str) -> Optional[Path]:
    files = sorted(glob.glob(str(CACHE_DIR / pattern)))
    return Path(files[-1]) if files else None

def load_rep_map(csv_path: Optional[Path]) -> dict:
    """cluster_to_rep_method.csv 로더: 헤더 'cluster,rep_method' 기준"""
    rep = {}
    if not (csv_path and csv_path.exists()):
        return rep
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                cid = int(row["cluster"])
                rep[cid] = row["rep_method"]
            except Exception:
                continue
    return rep

print("=== Artefacts Loading ===")
best_model_file = latest_file("best_lstm_model_*.pt")               # per-line 토큰 LSTM(폴백용)
label_enc_file  = latest_file("label_encoder_*.joblib")
kmeans_file     = latest_file("kmeans_*.joblib")
time_scaler_file= latest_file("time_scaler_*.joblib")
rep_method_csv  = REP_METHOD_CSV if REP_METHOD_CSV.exists() else latest_file("cluster_to_rep_method*.csv")

# 시퀀스 LSTM 가중치(학습 파이프라인의 D-입력 LSTM)
seq_model_file  = CACHE_DIR / "lstm_label_attack.pt"

print(f"per-line best_model : {best_model_file}")
print(f"label_enc           : {label_enc_file}")
print(f"kmeans              : {kmeans_file}")
print(f"time_scale          : {time_scaler_file}")
print(f"rep_method          : {rep_method_csv}")
print(f"seq_lstm (D-input)  : {seq_model_file if seq_model_file.exists() else 'MISSING'}")

if not all([label_enc_file, kmeans_file]):
    raise FileNotFoundError("필수 모델 파일(label_encoder, kmeans)이 누락되었습니다.")

# Tokenizer/Encoder
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
enc_model = AutoModel.from_pretrained(MODEL_NAME).eval().to(DEVICE)
H_dim = enc_model.config.hidden_size

# External artefacts
kmeans   = joblib.load(kmeans_file)
y_le     = joblib.load(label_enc_file)
scaler_t = joblib.load(time_scaler_file) if time_scaler_file else None
rep_map  = load_rep_map(rep_method_csv)
print(f"rep_map size: {len(rep_map)}")

K_clusters = int(kmeans.n_clusters)       # onehot K
C_classes  = len(y_le.classes_)           # label classes

# per-line 토큰 LSTM (폴백용)
token_lstm_model = None
if best_model_file and Path(best_model_file).exists():
    token_lstm_model = LSTMClassifier(input_h=H_dim, cluster_k=K_clusters, num_classes=C_classes).to(DEVICE)
    state = torch.load(best_model_file, map_location=DEVICE)
    token_lstm_model.load_state_dict(state)
    token_lstm_model.eval()

# 시퀀스 LSTM (학습과 동일한 D 입력)
D_input = H_dim + K_clusters + 1
seq_model = None
USE_SEQ_MODE = False
if seq_model_file.exists():
    try:
        seq_model = LSTMSeqClassifier(input_dim=D_input, num_classes=C_classes).to(DEVICE)
        seq_state = torch.load(seq_model_file, map_location=DEVICE)
        seq_model.load_state_dict(seq_state)
        seq_model.eval()
        USE_SEQ_MODE = True
        print("[SEQ] LSTMSeqClassifier loaded and enabled.")
    except Exception as e:
        print(f"[SEQ] Failed to load seq model: {e} (fallback to per-line)")
else:
    print("[SEQ] Sequence model file not found. Fallback to per-line.")

# 모델/토크나이저 동시 접근 보호
_model_lock = threading.Lock()

# =========================
# 예측 유틸 (공통)
# =========================
def _embed_payload(payload: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    반환:
      tok: (T, H_dim) float32
      msk: (T,) int64
      cls: (H_dim,) float32
    """
    enc = tokenizer([payload], padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = enc_model(**enc).last_hidden_state  # (1,T,H)
    tok = out[0].detach().cpu().numpy().astype(np.float32, copy=False)
    msk = enc["attention_mask"][0].detach().cpu().numpy().astype(np.int64)
    cls = out[:, 0, :].detach().cpu().numpy()[0].astype(np.float32, copy=False)
    tok = np.ascontiguousarray(tok, dtype=np.float32)
    msk = np.ascontiguousarray(msk, dtype=np.int64)
    cls = np.ascontiguousarray(cls, dtype=np.float32)
    return tok, msk, cls

def _cluster_id_from_cls(cls_vec: np.ndarray) -> int:
    try:
        return int(kmeans.predict(cls_vec.reshape(1, -1).astype(np.float32, copy=False))[0])
    except Exception:
        return int(kmeans.predict(cls_vec.reshape(1, -1).astype(np.float64, copy=False))[0])

def _onehot_cluster(idx: int, K: int) -> np.ndarray:
    v = np.zeros((K,), dtype=np.float32)
    if 0 <= idx < K:
        v[idx] = 1.0
    return v

def _dt_feature(prev_ts: Optional[float], cur_ts: float) -> float:
    dt = 0.0 if prev_ts is None else max(0.0, cur_ts - prev_ts)
    raw = np.log1p(dt).astype(np.float32).reshape(1,1)
    if scaler_t is not None:
        return float(np.float32(scaler_t.transform(raw.astype(np.float64))[0,0]))
    return float(raw[0,0])

# =========================
# per-line (폴백) 예측
# =========================
def predict_per_line(payload: str, time_seconds: float = 0.0):
    """
    payload 하나 → (cluster, method, label, p_attack, p_normal)
    per-line 토큰 LSTM 사용
    """
    if token_lstm_model is None:
        # 시퀀스 모델도, 토큰 모델도 없으면 예측 불가
        raise RuntimeError("No per-line LSTM model available.")
    with _model_lock:
        tok_np, mask_np, cls_vec = _embed_payload(payload)
        clu_id = _cluster_id_from_cls(cls_vec)
        method_pred = rep_map.get(clu_id, f"CLUSTER_{clu_id}")
        # 시간 특성
        tfeat_raw = np.log1p(float(time_seconds)).astype(np.float32).reshape(1, 1)
        if scaler_t is not None:
            t_scaled64 = scaler_t.transform(tfeat_raw.astype(np.float64, copy=False))
            tfeat_scaled = np.float32(t_scaled64.reshape(-1)[0])
        else:
            tfeat_scaled = np.float32(tfeat_raw.reshape(-1)[0])

        # Torch 텐서 변환
        toks  = torch.from_numpy(tok_np[np.newaxis, :, :]).to(dtype=torch.float32, device=DEVICE)
        masks = torch.from_numpy(mask_np[np.newaxis, :]).to(dtype=torch.int64,   device=DEVICE)
        clus  = torch.tensor([clu_id], dtype=torch.long,    device=DEVICE)
        tft   = torch.tensor([tfeat_scaled], dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
            logits = token_lstm_model(toks, masks, clus, tft)
            probs  = torch.softmax(logits, dim=1).detach().cpu().numpy()[0].astype(np.float32, copy=False)

        pred_i = int(np.argmax(probs))
        pred_label = y_le.inverse_transform([pred_i])[0]
        prob_map = {lab: float(probs[i]) for i, lab in enumerate(y_le.classes_)}
        prob_attack = float(prob_map.get("attack", probs[0] if len(probs) > 0 else 0.0))
        prob_normal = float(prob_map.get("normal", probs[1] if len(probs) > 1 else 0.0))

        return clu_id, method_pred, pred_label, prob_attack, prob_normal

# =========================
# 시퀀스 추론 버퍼 & 예측
# =========================
from collections import deque, defaultdict
_seq_buffers = defaultdict(lambda: deque(maxlen=SEQ_LEN))
_prev_ts     = {}   # 키별 이전 epoch seconds

def _make_step_feature_from_payload(payload: str, clu_id: int, dt_feat: float) -> np.ndarray:
    """
    한 타임스텝의 특징: [CLS(768) || onehot(K) || dt(1)] -> shape (D_input,)
    """
    with torch.no_grad():
        enc = tokenizer([payload], padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
        cls = enc_model(**enc).last_hidden_state[:,0,:].detach().cpu().numpy()[0].astype(np.float32, copy=False)
    oh = _onehot_cluster(clu_id, K_clusters)
    return np.concatenate([cls, oh, np.array([dt_feat], dtype=np.float32)], axis=0)

def predict_sequence_for_key(key: str):
    """
    버퍼가 SEQ_LEN에 도달하면 [1, T, D]로 묶어 예측.
    반환: (pred_label, p_attack, p_normal) | None
    """
    if not USE_SEQ_MODE or seq_model is None:
        return None
    buf = _seq_buffers[key]
    if len(buf) < SEQ_LEN:
        return None
    X = np.stack(list(buf), axis=0).astype(np.float32)      # [T, D]
    xt = torch.from_numpy(X[np.newaxis, :, :]).to(DEVICE)   # [1, T, D]
    with torch.no_grad():
        logits = seq_model(xt)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0].astype(np.float32)
    pred_i = int(np.argmax(probs))
    pred_label = y_le.inverse_transform([pred_i])[0]
    prob_map = {lab: float(probs[i]) for i, lab in enumerate(y_le.classes_)}
    p_attack = float(prob_map.get("attack", probs[0] if len(probs)>0 else 0.0))
    p_normal = float(prob_map.get("normal", probs[1] if len(probs)>1 else 0.0))
    return pred_label, p_attack, p_normal

# =========================
# CSV 유틸
# =========================
_csv_lock = threading.Lock()

def ensure_csv_header():
    with _csv_lock:
        need = (not OUT_CSV.exists()) or (OUT_CSV.stat().st_size == 0)
        if need:
            with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_HEADER)

def append_csv_row(row: list, dry_run: bool = False):
    if dry_run:
        return
    with _csv_lock:
        with OUT_CSV.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

# =========================
# 공용 처리 함수: 한 줄 처리 (시퀀스 우선, 폴백 지원)
# =========================
class ThreadedHandler(socketserver.StreamRequestHandler):
    raw_lock = threading.Lock()  # 원문 로그 파일 쓰기 락

def process_one_line(line: str, peer_ip: str = "", user: str = "file",
                     save_raw: bool = True, dry_run: bool = False):
    """접수된 라인(서버/파일 공통)을 파싱+예측+CSV 기록 (시퀀스 우선)"""
    # 타임스탬프(원문 저장용)
    try:
        timestamp_iso = datetime.datetime.now(datetime.UTC).isoformat()
    except Exception:
        timestamp_iso = datetime.datetime.utcnow().isoformat() + "Z"

    # 원문 로그 저장(옵션)
    if save_raw:
        try:
            with ThreadedHandler.raw_lock:
                OUT_RAW_LOG.parent.mkdir(parents=True, exist_ok=True)
                with OUT_RAW_LOG.open('a', encoding='utf-8') as f:
                    f.write(f"{timestamp_iso} {peer_ip} {user} {line}\n")
        except Exception as e:
            print(f"[-] Failed to write raw log: {e}")

    # 파싱
    data = parse_apache_line(line) or {}
    req  = data.get("request")
    payload = extract_payload_from_request(req, line)
    key = data.get("ip") or peer_ip or "unknown"

    # 시간 파싱 → Δt
    cur_ts = parse_apache_time_to_epoch(data.get("time"))
    dt_feat = _dt_feature(_prev_ts.get(key), cur_ts)
    _prev_ts[key] = cur_ts

    # 클러스터 ID (CLS 기반 KMeans)
    with _model_lock:
        _, _, cls_vec = _embed_payload(payload)
    clu_id = _cluster_id_from_cls(cls_vec)
    method_pred = rep_map.get(clu_id, f"CLUSTER_{clu_id}")

    # 시퀀스 스텝 피처 생성 → 버퍼 push
    step = _make_step_feature_from_payload(payload, clu_id, dt_feat)  # (D_input,)
    _seq_buffers[key].append(step)

    # 시퀀스 모델 사용 가능 시: 버퍼 길이 충족하면 예측
    pred_label = None
    p_attack = p_normal = None
    used_seq = False
    if USE_SEQ_MODE:
        out = predict_sequence_for_key(key)
        if out is not None:
            pred_label, p_attack, p_normal = out
            used_seq = True

    # 시퀀스 예측이 아직 불가(버퍼 부족 등)하면 per-line 폴백
    if pred_label is None:
        try:
            clu_id2, method_pred2, pred_label2, p_attack2, p_normal2 = predict_per_line(payload, time_seconds=0.0)
            # per-line에서 나온 method와 cluster를 사용(일관성 위해 method_pred 유지 가능)
            pred_label = pred_label2
            if method_pred == f"CLUSTER_{clu_id}":  # rep_map 없을 때만 대체
                method_pred = method_pred2
            p_attack, p_normal = p_attack2, p_normal2
        except Exception as e:
            print(f"[-] predict fallback error: {e}")
            return

    # CSV 기록
    row = [
        data.get("time", ""),
        data.get("ip", ""),
        data.get("request", ""),
        data.get("status", ""),
        data.get("size", ""),
        clu_id,
        method_pred,
        pred_label,
        f"{p_attack:.6f}",
        f"{p_normal:.6f}",
    ]
    append_csv_row(row, dry_run=dry_run)

    # 콘솔 출력
    mode_tag = "SEQ" if used_seq else "LINE"
    print(f"[{data.get('time','')}] ({mode_tag}:{len(_seq_buffers[key])}) {payload} "
          f"-> {method_pred} | {pred_label} (attack={p_attack:.3f}, normal={p_normal:.3f})")

# =========================
# TCP 핸들러 (서버 모드)
# =========================
class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True

class TCPHandler(StreamRequestHandler:=socketserver.StreamRequestHandler):
    # 별칭으로 남기고 싶었으나 일부 파서가 싫어할 수 있어 표준 클래스 그대로 사용합니다.
    pass

class ThreadedHandler(socketserver.StreamRequestHandler):
    raw_lock = threading.Lock()  # 원문 로그 파일 쓰기 락

    def handle(self):
        peer = self.client_address[0]

        # --- 1) 인증: 첫 줄 JSON {user, pass}
        try:
            raw = self.rfile.readline()
            if not raw:
                return
            try:
                auth = json.loads(raw.decode('utf-8').strip())
            except Exception:
                self.wfile.write(b'ERR invalid auth format\n')
                return
            user = auth.get("user")
            passwd = auth.get("pass")
            if user != EXPECTED_USER or passwd != EXPECTED_PASS:
                self.wfile.write(b'ERR auth failed\n')
                return
            self.wfile.write(b'OK\n')
            print(f"[+] {peer} authenticated as {user}")
        except Exception as e:
            print(f"[-] auth error from {peer}: {e}")
            return

        # --- 2) 라인 수신 → 공용 처리(시퀀스 우선)
        for raw_line in self.rfile:
            try:
                line = raw_line.decode('utf-8', errors='replace').rstrip('\r\n')
            except Exception:
                line = raw_line.decode('latin1', errors='replace').rstrip('\r\n')
            process_one_line(line, peer_ip=peer, user=user, save_raw=True, dry_run=False)

# =========================
# 파일 모드 (이전 로그 일괄 분석)
# =========================
def run_file_mode(file_path: Path, encoding: str = "utf-8", save_raw: bool = True, dry_run: bool = False):
    if not file_path.exists():
        raise FileNotFoundError(f"입력 로그 파일이 존재하지 않습니다: {file_path}")
    print(f"[+] File mode: {file_path} (encoding={encoding})")
    ensure_csv_header()

    # 인코딩 폴백
    try:
        with file_path.open("r", encoding=encoding, errors="strict") as f:
            for line in f:
                process_one_line(line.rstrip("\r\n"), peer_ip="file", user="file", save_raw=save_raw, dry_run=dry_run)
    except UnicodeDecodeError:
        print("[!] utf-8 디코딩 실패 → latin1로 재시도")
        with file_path.open("r", encoding="latin1", errors="replace") as f:
            for line in f:
                process_one_line(line.rstrip("\r\n"), peer_ip="file", user="file", save_raw=save_raw, dry_run=dry_run)

# =========================
# main
# =========================
def main():
    parser = argparse.ArgumentParser(description="Auth Collect & Classify (Sequence/Line, Server/File)")
    parser.add_argument("--from-file", type=str, default=None, help="과거 Apache 액세스 로그 파일 경로")
    parser.add_argument("--encoding",  type=str, default="utf-8", help="파일 인코딩 (기본 utf-8)")
    parser.add_argument("--no-raw",    action="store_true", help="원문 로그(collected_access.log) 저장하지 않음")
    parser.add_argument("--dry-run",   action="store_true", help="CSV에 기록하지 않고 콘솔만 출력")
    args = parser.parse_args()

    ensure_csv_header()

    if args.from_file:
        run_file_mode(Path(args.from_file), encoding=args.encoding, save_raw=(not args.no_raw), dry_run=args.dry_run)
        return

    # 서버 모드
    srv = ThreadedTCPServer((LISTEN_HOST, LISTEN_PORT), ThreadedHandler)
    print(f"[+] Auth+Classify server on {LISTEN_HOST}:{LISTEN_PORT}")
    print(f"    raw -> {OUT_RAW_LOG}")
    print(f"    csv -> {OUT_CSV}")
    print(f"    mode -> {'SEQUENCE' if USE_SEQ_MODE else 'PER-LINE (fallback)'}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down ...")
        srv.shutdown()
        srv.server_close()

if __name__ == "__main__":
    main()
