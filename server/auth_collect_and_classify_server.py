#!/usr/bin/env python3
# auth_collect_and_classify_server.py
# -*- coding: utf-8 -*-
"""
하나로 합치기:
- TCP 인증 수신 서버: 로그 라인 수신 → collected_access.log 저장
- 즉시 분류: 페이로드 추출 → 탐지명 + 위험라벨 예측 → detections.csv 저장

실행:
    python auth_collect_and_classify_server.py
"""

# ===== 환경 변수 (토크나이저 관련 이슈 회피) =====
import os
os.environ["TRANSFORMERS_USE_FAST"] = "0"               # fast tokenizer 강제 비활성
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"  # C 확장 대신 python 구현 사용

# ===== 기본 모듈 =====
import socketserver
import threading
import json
import datetime
import re
import csv
import glob
from pathlib import Path

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
EXPECTED_PASS = "user1234"   # 테스트용 평문 (운영 환경에서는 반드시 TLS/토큰 등으로 교체)

# --- 파일 경로 ---
OUT_RAW_LOG   = Path("./collected_access.log")   # 수신 원문 로그 저장
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


# =========================
# Apache 로그 파서 & 페이로드 추출
# =========================
APACHE_RE = re.compile(
    r'(?P<ip>\S+)\s+\S+\s+\S+\s+\[(?P<time>[^\]]+)\]\s+'
    r'"(?P<request>[^"]*)"\s+(?P<status>\d{3})\s+(?P<size>\S+)'
    r'(?:\s+"(?P<referer>[^"]*)"\s+"(?P<agent>[^"]*)")?'
)

def parse_apache_line(line: str):
    m = APACHE_RE.match(line)
    if not m:
        return None
    return m.groupdict()

def extract_payload_from_request(req: str, fallback_line: str):
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
# LSTM 모델 정의
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
# Artefact 로드 (서버 시작 시 1회)
# =========================
def latest_file(pattern: str):
    files = sorted(glob.glob(str(CACHE_DIR / pattern)))
    return Path(files[-1]) if files else None

def load_rep_map(csv_path: Path) -> dict:
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
best_model_file = latest_file("best_lstm_model_*.pt")
label_enc_file  = latest_file("label_encoder_*.joblib")
kmeans_file     = latest_file("kmeans_*.joblib")
time_scaler_file= latest_file("time_scaler_*.joblib")

# rep_method.csv는 명시 경로 우선, 없으면 CACHE_DIR에서 자동 탐색
rep_method_csv  = REP_METHOD_CSV if REP_METHOD_CSV.exists() else latest_file("cluster_to_rep_method*.csv")

print(f"best_model : {best_model_file}")
print(f"label_enc  : {label_enc_file}")
print(f"kmeans     : {kmeans_file}")
print(f"time_scale : {time_scaler_file}")
print(f"rep_method : {rep_method_csv}")

if not all([best_model_file, label_enc_file, kmeans_file]):
    raise FileNotFoundError("필수 모델 파일이 누락되었습니다.")

# Tokenizer/Encoder
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
enc_model = AutoModel.from_pretrained(MODEL_NAME).eval().to(DEVICE)
H_dim = enc_model.config.hidden_size

# External artefacts
kmeans   = joblib.load(kmeans_file)
y_le     = joblib.load(label_enc_file)
scaler_t = joblib.load(time_scaler_file) if time_scaler_file else None

# 대표 탐지명 매핑 로더 적용
rep_map = load_rep_map(rep_method_csv)
print(f"rep_map size: {len(rep_map)}")

cluster_k   = int(kmeans.n_clusters)
num_classes = len(y_le.classes_)

# 모델 로드
model = LSTMClassifier(input_h=H_dim, cluster_k=cluster_k, num_classes=num_classes).to(DEVICE)
state = torch.load(best_model_file, map_location=DEVICE)
model.load_state_dict(state)
model.eval()

# 모델/토크나이저 동시 접근 보호
_model_lock = threading.Lock()


# =========================
# 예측 함수 (thread-safe)
# =========================
def _embed_payload(payload: str):
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

def predict(payload: str, time_seconds: float = 0.0):
    try:
        with _model_lock:
            tok_np, mask_np, cls_vec = _embed_payload(payload)
            tok_np  = np.ascontiguousarray(tok_np,  dtype=np.float32)
            mask_np = np.ascontiguousarray(mask_np, dtype=np.int64)
            cls_vec = np.ascontiguousarray(cls_vec, dtype=np.float32)

            # KMeans 예측: float32 시도 → 실패 시 float64 재시도
            try:
                clu_id = int(kmeans.predict(cls_vec.reshape(1, -1).astype(np.float32, copy=False))[0])
            except Exception:
                clu_id = int(kmeans.predict(cls_vec.reshape(1, -1).astype(np.float64, copy=False))[0])

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
                logits = model(toks, masks, clus, tft)
                probs  = torch.softmax(logits, dim=1).detach().cpu().numpy()[0].astype(np.float32, copy=False)

            pred_i = int(np.argmax(probs))
            pred_label = y_le.inverse_transform([pred_i])[0]

            prob_map = {lab: float(probs[i]) for i, lab in enumerate(y_le.classes_)}
            prob_attack = float(prob_map.get("attack", probs[0] if len(probs) > 0 else 0.0))
            prob_normal = float(prob_map.get("normal", probs[1] if len(probs) > 1 else 0.0))

            return clu_id, method_pred, pred_label, prob_attack, prob_normal
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise


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

def append_csv_row(row: list):
    with _csv_lock:
        with OUT_CSV.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)


# =========================
# TCP 핸들러
# =========================
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

        # --- 2) 라인 수신 → 파일 저장 + 즉시 분류 + CSV 기록
        for raw_line in self.rfile:
            try:
                line = raw_line.decode('utf-8', errors='replace').rstrip('\r\n')
            except Exception:
                line = raw_line.decode('latin1', errors='replace').rstrip('\r\n')

            timestamp = datetime.datetime.now(datetime.UTC).isoformat()
            raw_out = f"{timestamp} {peer} {user} {line}"

            # 2-1) 원문 로그 저장
            try:
                with self.raw_lock:
                    OUT_RAW_LOG.parent.mkdir(parents=True, exist_ok=True)
                    with OUT_RAW_LOG.open('a', encoding='utf-8') as f:
                        f.write(raw_out + "\n")
            except Exception as e:
                print(f"[-] Failed to write raw log: {e}")

            # 2-2) 파싱 & 페이로드 생성
            data = parse_apache_line(line) or {}
            req  = data.get("request")
            payload = extract_payload_from_request(req, line)

            # 2-3) 예측
            try:
                clu, method, label, p_attack, p_normal = predict(payload, time_seconds=0.0)
            except Exception as e:
                print(f"[-] predict error: {e}")
                continue

            # 2-4) CSV 기록
            row = [
                data.get("time", ""),
                data.get("ip", ""),
                data.get("request", ""),
                data.get("status", ""),
                data.get("size", ""),
                clu,
                method,
                label,
                f"{p_attack:.6f}",
                f"{p_normal:.6f}",
            ]
            append_csv_row(row)

            # 콘솔 출력
            print(f"[{data.get('time','')}] {payload} -> {method} | {label} "
                  f"(attack={p_attack:.3f}, normal={p_normal:.3f})")


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True


# =========================
# main
# =========================
def main():
    ensure_csv_header()
    srv = ThreadedTCPServer((LISTEN_HOST, LISTEN_PORT), ThreadedHandler)
    print(f"[+] Auth+Classify server on {LISTEN_HOST}:{LISTEN_PORT}")
    print(f"    raw -> {OUT_RAW_LOG}")
    print(f"    csv -> {OUT_CSV}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down ...")
        srv.shutdown()
        srv.server_close()

if __name__ == "__main__":
    main()
