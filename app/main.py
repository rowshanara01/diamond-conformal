"""Appraisal API: hirer dam + 90% GUARANTEED-style range (conformal) + asking price judge kora"""
import json
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

import common as C

B = joblib.load("model/appraiser.joblib")
app = FastAPI(title="Diamond Appraisal API")


class Diamond(BaseModel):
    carat: float = Field(ge=0.2, le=5.01)
    cut: Literal["Fair", "Good", "Very Good", "Premium", "Ideal"]
    color: Literal["J", "I", "H", "G", "F", "E", "D"]
    clarity: Literal["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
    depth: float = Field(ge=40, le=80)
    table: float = Field(ge=40, le=80)
    asking_price: Optional[float] = Field(default=None, gt=0, description="USD, thakle range er sathe tulona")


def appraise(d: Diamond):
    row = C.encode(pd.DataFrame([d.model_dump(exclude={"asking_price"})]))[C.FEATURES]
    mu = float(B["mean"].predict(row)[0])
    lo = min(float(B["lo"].predict(row)[0]) - B["qhat"], mu)      # conformal: calibration theke paoa qhat diye boro kora
    hi = max(float(B["hi"].predict(row)[0]) + B["qhat"], mu)
    est, low, high = np.exp(mu), np.exp(lo), np.exp(hi)
    out = {"estimated_price": round(float(est)), "range_low": round(float(low)), "range_high": round(float(high)),
           "interval_level": 1 - C.ALPHA, "warnings": []}
    if d.carat > B["ranges"]["carat_p99"]:
        out["warnings"].append(f"carat {B['ranges']['carat_p99']:.2f} er upore data onek kom (matro ~1%), range-er upor bhorosha kom")
    if d.asking_price is not None:
        out["asking_vs_estimate_pct"] = round(100 * (d.asking_price / est - 1), 1)
        out["verdict"] = ("below_range" if d.asking_price < low else "above_range" if d.asking_price > high else "within_range")
    return out


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return json.load(open("model/metrics.json"))


@app.post("/appraise")
def appraise_one(d: Diamond):
    return appraise(d)


@app.post("/appraise-batch")
def appraise_batch(ds: list[Diamond]):
    return {"count": len(ds), "results": [appraise(d) for d in ds]}
