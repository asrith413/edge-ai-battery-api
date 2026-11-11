"""
app/main.py -- STARTER battery API. Look for TODO.

    uvicorn app.main:app --host 0.0.0.0 --port 8000

GET  /health          -> {"status": "ok", "team": 7, "params": 1281, "val_mae": 3.4}
GET  /model_info      -> the per-layer table: params, bytes, MACs, and whether it fits the doorbell
POST /predict         one reading  -> {"days_left": 52.1, "timings_ms": {...}}
POST /predict_batch   {"rows": [reading, ...]} -> {"days_left": [...], "timings_ms": {...}}
A reading that is out of range gets HTTP 422 with the reason, never a guess.
"""
import json
import time

import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from app.model import FEATURES, BatteryNet

cfg = json.load(open("model/config.json"))
net = BatteryNet(tuple(cfg["hidden"]))
net.load_state_dict(torch.load("model/model.pt", weights_only=True))
net.eval()
torch.set_num_threads(1)
MU, SD = np.array(cfg["mu"], np.float32), np.array(cfg["sd"], np.float32)
N_PARAMS = sum(p.numel() for p in net.parameters())


class Reading(BaseModel):
    # Mission 3: ranges a real sensor can physically report, deliberately wider than our data
    # (so a genuine reading just outside the training range still gets an answer), but still
    # rejecting the Mission-1 glitches (battery 0 V, temp -127, RSSI 0) with HTTP 422.
    battery_v: float = Field(ge=2.0, le=4.5)         # Li-ion cell; data spans 3.0-4.2
    temp_c: float = Field(ge=-20.0, le=60.0)         # indoor/outdoor doorbell; data 8-46
    rings_per_day: float = Field(ge=0.0, le=200.0)   # a count cannot be negative; data 0-35
    wifi_rssi_dbm: float = Field(ge=-100.0, le=-20.0)  # dBm is negative; 0 = no reading; data -88..-46
    age_months: float = Field(ge=0.0, le=120.0)      # 0-10 years installed; data 0-36


class Batch(BaseModel):
    rows: list[Reading] = Field(min_length=1, max_length=10000)


app = FastAPI(title="Doorbell battery API")
ms = lambda t0: round((time.perf_counter() - t0) * 1000, 3)


@app.get("/model_info")
def model_info():
    # Mission 2 + 4: build the per-layer table FROM THE NETWORK (net.net), not by hand, so it
    # stays correct if the hidden sizes change. For a Linear(a, b): a*b weights + b biases, so
    # params = a*b + b; MACs per prediction = a*b; float32 = 4 bytes/param, int8 = 1 byte/param.
    # ReLU layers carry no parameters, so we skip anything that is not a Linear.
    layers, total_params, total_macs, n = [], 0, 0, 0
    for m in net.net:
        if not isinstance(m, torch.nn.Linear):
            continue
        n += 1
        p = m.in_features * m.out_features + m.out_features
        macs = m.in_features * m.out_features
        layers.append({
            "name": f"linear{n}",
            "in": m.in_features,
            "out": m.out_features,
            "params": p,
            "bytes_f32": p * 4,
            "macs": macs,
        })
        total_params += p
        total_macs += macs
    total_bytes_f32 = total_params * 4
    total_bytes_int8 = total_params * 1
    return {
        "layers": layers,
        "total_params": total_params,
        "total_bytes_f32": total_bytes_f32,
        "total_bytes_int8": total_bytes_int8,
        "total_macs": total_macs,
        "fits_300kb_sram_f32": total_bytes_f32 <= 300 * 1024,
        "fits_300kb_sram_int8": total_bytes_int8 <= 300 * 1024,
    }


@app.get("/health")
def health():
    return {"status": "ok", "team": cfg["team"], "params": N_PARAMS, "val_mae": cfg["val_mae"]}


async def run(request: Request, schema):
    t_all = time.perf_counter(); t = {}
    t0 = time.perf_counter()
    try:
        data = schema.model_validate_json(await request.body())
    except ValidationError as e:
        return JSONResponse(status_code=422, content={"error": "bad reading", "detail": json.loads(e.json())})
    t["parse"] = ms(t0)
    rows = data.rows if isinstance(data, Batch) else [data]

    # Mission 3: time the remaining three stages exactly like "parse" above.
    t0 = time.perf_counter()
    x = (np.array([[getattr(r, f) for f in FEATURES] for r in rows], np.float32) - MU) / SD
    t["normalise"] = ms(t0)

    t0 = time.perf_counter()
    with torch.no_grad():
        y = net(torch.from_numpy(x)).numpy()
    t["inference"] = ms(t0)

    t0 = time.perf_counter()
    days = [round(float(v), 2) for v in np.maximum(y, 0)]
    body = {"days_left": days if isinstance(data, Batch) else days[0]}
    t["build"] = ms(t0)

    t["server_total"] = ms(t_all)
    body["timings_ms"] = t
    return body


@app.post("/predict")
async def predict(request: Request):
    return await run(request, Reading)


@app.post("/predict_batch")
async def predict_batch(request: Request):
    return await run(request, Batch)
