# Diamond Pricing with GUARANTEED Prediction Intervals (REAL data)

Data: 53,940 ashol hirer record (seaborn-data, diamonds.csv). Model sudhu "dam koto" na, bole **"90% shomoy ashol dam ei range e thakbe"**.

## Chalanor niyom
pip install -r requirements.txt
python train.py          # clean -> models -> monotonic -> conformal interval -> segment coverage
pytest -q
uvicorn app.main:app --reload     # POST /appraise  (/docs)
(data na thakle: python download_data.py)

## Ja ja ache (high-level)
1. Real data cleaning: dimension 0 / 58.9mm typo, 145ta duplicate row (dedupe na korle test row er ~0.5% er twin train e thake)
2. Log-price modeling, baseline (price/carat) vs linear vs gradient boosting
3. MONOTONIC constraint: carat/quality barle dam komte parbe na (domain knowledge model e dhukano). Tar dam: accuracy ektu kome (5.8% -> 6.5% median error)
4. CONFORMAL PREDICTION (Conformalized Quantile Regression): calibration set theke 'koto bhul hoy' mepe interval boro kora.
   Raw quantile regression 87% cover kore (lokkho 90%), conformal ta thik 90%
5. Segment-wise coverage audit: overall 90% mane shob group e 90% na. Fair cut e ~82% (nicher section dekho)
6. Train / Calibration / Test 3-way split (calibration data model train e use hoy na)
7. API: estimate + range + asking price verdict + sparse-region warning (carat beshi hole data kom)

## Sotti kotha (limitation)
- Conformal er guarantee **marginal**: shob hire milie 90%. Kono ekta group (jemon Fair cut) e kom hote pare.
- Guarantee shudhu tokhon, jokhon notun data train data r moto-i (exchangeability). Notun bazar/shomoy hole bhenge jay (drift).
- "above_range" mane na je overpriced; hoyto special feature (certificate, fluorescence) model e nai.
- Dimension (x,y,z), certificate, fluorescence, shomoy model e nai.

## Challenge
1. Mondrian/class-conditional conformal: prottek cut e alada 90% guarantee
2. Normalized conformal (interval width difficulty onujayi)
3. Adaptive conformal inference (shomoy bodlale coverage ta thik rakha)
4. SHAP/ICE plot diye price vs carat dekho
5. Price drift monitoring (fraud-mlops er monitor.py reuse)
