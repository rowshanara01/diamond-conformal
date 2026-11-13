# Diamond Pricing with Guaranteed Prediction Intervals (Real Data) #

**Data:** 53,940 real diamond records (seaborn-data, `diamonds.csv`).

The model doesn't just say "how much it costs." It says:
**"90% of the time, the true price will fall inside this range."**

## How to run #

```bash
pip install -r requirements.txt
python train.py                   # clean -> models -> monotonic -> conformal interval -> segment coverage
pytest -q
uvicorn app.main:app --reload     # POST /appraise  (docs at /docs)
```

If the data is missing, run:

```bash
python download_data.py
```

## What's included ##

1. **Real data cleaning**
   Fixes zero-dimension rows and the 58.9 mm typo, and removes 145 duplicate rows. Without deduplication, roughly 0.5% of test rows would have an identical twin in the training set.
2. **Log-price modeling**
   Compares a baseline (price per carat), a linear model, and gradient boosting.
3. **Monotonic constraints**
   A higher carat or better quality can never lower the predicted price. This builds domain knowledge into the model. The cost is a small drop in accuracy (median error goes from 5.8% to 6.5%).
4. **Conformal prediction (Conformalized Quantile Regression)**
   Uses a calibration set to measure how wrong the model typically is, then widens the interval accordingly. Raw quantile regression covers only 87% (target: 90%), while the conformal version hits 90%.
5. **Segment-wise coverage audit**
   90% overall does not mean 90% in every group. For example, Fair-cut diamonds get about 82% (see the audit output).
6. **Train / calibration / test three-way split**
   Calibration data is never used to train the model.
7. **API**
   Returns a point estimate, a price range, a verdict on the asking price, and a sparse-region warning (data is thin for large carat sizes).

## Limitations ##

- **The guarantee is marginal.** Coverage is 90% across all diamonds combined, but a specific group (such as Fair cut) can fall short.
- **Exchangeability is required.** The guarantee only holds when new data looks like the training data. It breaks under a new market or over time (drift).
- **"Above range" does not mean overpriced.** A special feature (certificate, fluorescence) that the model doesn't know about could explain the gap.
- **Missing inputs.** The model does not use the dimensions (x, y, z), certificate, fluorescence, or time.

## Challenges / next steps

1. **Mondrian (class-conditional) conformal:** a separate 90% guarantee for each cut.
2. **Normalized conformal:** interval width adapts to how difficult each prediction is.
3. **Adaptive conformal inference:** keeps coverage correct when the data changes over time.
4. **SHAP / ICE plots:** examine how price varies with carat.
5. **Price drift monitoring:** reuse `monitor.py` from the fraud-mlops project.
