import json

from fastapi.testclient import TestClient

import common as C
from app.main import app

client = TestClient(app)
M = json.load(open("model/metrics.json"))
BASE = {"carat": 1.0, "cut": "Very Good", "color": "G", "clarity": "VS2", "depth": 61.5, "table": 57}


def est(**kw):
    return client.post("/appraise", json={**BASE, **kw}).json()["estimated_price"]


def test_cleaning_removed_bad_rows_and_duplicates():
    df, rep = C.load()
    assert rep["duplicates_removed"] > 100 and rep["dropped_bad_values"] > 0 and not df.duplicated().any()
    assert df.carat.min() > 0 and (df.price > 0).all()


def test_point_model_accuracy_gates():
    m = M["models"]
    assert m["hgb_monotonic"]["median_ape"] < 0.08 and m["baseline_price_per_carat"]["median_ape"] > 0.25
    assert m["hgb_monotonic"]["median_ape"] < m["linear_log"]["median_ape"]


def test_conformal_coverage_close_to_90_and_better_than_raw_quantiles():
    c = M["coverage"]
    assert 0.88 <= c["conformal_CQR"]["overall"] <= 0.92
    assert c["conformal_CQR"]["overall"] > c["raw_quantile_regression"]["overall"] + 0.01
    assert c["conformal_CQR"]["median_width_ratio"] <= c["normal_residual (+-1.645 sd)"]["median_width_ratio"]


def test_monotonic_constraint_enforced():
    assert M["monotonic_violation_rate"]["monotonic"] == 0.0


def test_api_price_nondecreasing_in_carat_and_quality():
    carats = [round(0.3 + 0.1 * i, 1) for i in range(23)]
    prices = [est(carat=c) for c in carats]
    assert all(a <= b for a, b in zip(prices, prices[1:]))
    assert est(cut="Ideal") >= est(cut="Fair") and est(color="D") >= est(color="J") and est(clarity="IF") >= est(clarity="I1")


def test_interval_order_and_verdicts():
    r = client.post("/appraise", json=BASE).json()
    assert r["range_low"] < r["estimated_price"] < r["range_high"] and r["interval_level"] == 0.9
    assert client.post("/appraise", json={**BASE, "asking_price": r["estimated_price"] * 10}).json()["verdict"] == "above_range"
    assert client.post("/appraise", json={**BASE, "asking_price": r["estimated_price"] * 0.1}).json()["verdict"] == "below_range"
    ok = client.post("/appraise", json={**BASE, "asking_price": r["estimated_price"]}).json()
    assert ok["verdict"] == "within_range" and ok["asking_vs_estimate_pct"] == 0.0


def test_sparse_region_warning_and_validation():
    assert client.post("/appraise", json={**BASE, "carat": 4.0}).json()["warnings"]
    assert not client.post("/appraise", json=BASE).json()["warnings"]
    assert client.post("/appraise", json={**BASE, "carat": 0}).status_code == 422
    assert client.post("/appraise", json={**BASE, "cut": "Amazing"}).status_code == 422
