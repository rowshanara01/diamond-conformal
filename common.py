import numpy as np
import pandas as pd

ORD = {"cut": ["Fair", "Good", "Very Good", "Premium", "Ideal"],
       "color": ["J", "I", "H", "G", "F", "E", "D"],                       # D shobcheye bhalo
       "clarity": ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]}
FEATURES = ["carat", "cut_rank", "color_rank", "clarity_rank", "depth", "table"]
MONO = [1, 1, 1, 1, 0, 0]   # carat o quality barle dam KOMBE NA (baki sob same thakle)
ALPHA = 0.10                # 90% prediction interval
# x, y, z (dimension) model e nei: listing er shomoy jeweler jane carat/cut/color/clarity/depth/table; dimension carat er sathe prai ek


def load(path="data/diamonds.csv"):
    raw = pd.read_csv(path)
    rep = {"raw_rows": len(raw)}
    bad_dim = (raw[["x", "y", "z"]] == 0).any(axis=1) | (raw.y > 30) | (raw.z > 30)       # dimension 0 ba 58.9mm (typo)
    bad_geo = ~raw.depth.between(40, 80) | ~raw.table.between(40, 80)
    df = raw[~bad_dim & ~bad_geo]
    rep["dropped_bad_values"] = int(len(raw) - len(df))
    d = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)   # duplicate row train + test duiteyi gele leakage hoy
    rep["duplicates_removed"] = d
    rep["clean_rows"] = len(df)
    return encode(df), rep


def encode(df):
    out = df.copy()
    for col in ORD:
        out[f"{col}_rank"] = out[col].map({v: i for i, v in enumerate(ORD[col])})
    return out
