# Project Charter: Dynamic Hotel Overbooking Optimization (`docs/CHARTER.md`)

## 1. Business Problem & Objective
Hotels lose substantial revenue to empty rooms caused by last-minute booking cancellations, yet overselling too aggressively forces front-desk staff to "walk" confirmed guests to partner properties—incurring direct relocation fees and severe brand damage.

* **Client:** Two-property hotel group (`City Hotel` and `Resort Hotel`).
* **Objective:** Replace the client's static flat-rate overbooking heuristic with a data-driven, day-of-booking cancellation risk model and uncertainty-aware room recommendation policy.
* **Scope:** Predict cancellation probabilities using **strictly Bin-1 (fixed at booking)** attributes and convert them into integer nightly oversell counts $k$ for each `(hotel, arrival_date)`.

---

## 2. Success Metrics & Operational Guardrails (`src/contract.py`)

All policies are governed by the client's deterministic financial and brand-risk contract:

1. **Primary Financial Objective — Minimize Total Error Bill (`cost_eur`):**
   $$\text{Total Cost (€)} = \sum_{t} \left( \text{Empty Rooms}_t \times \text{Stay Value}_t + \text{Walked Guests}_t \times €300 \right)$$
   * **Empty Room Opportunity Cost:** $\max(0, \text{Cancelled}_t - k_t) \times \text{mean}(\text{ADR} \times \text{Nights})_t$.
   * **Walk Penalty:** $\max(0, k_t - \text{Cancelled}_t) \times €300.00$ per walked guest.

2. **Hard Brand Guardrail — Walk Ceiling (`WALK_CEILING <= 10.0`):**
   $$\text{Walks per 1,000 Arrivals} = 1000 \times \frac{\sum_t \text{Walked Guests}_t}{\sum_t (\text{Book}_t - \text{Cancelled}_t)} \le 10.0$$
   * Any policy exceeding $10.0$ walks per $1,000$ actual arrivals on the `Choose` selection window is disqualified.

3. **Secondary Diagnostic Metrics:**
   * **Discrimination:** ROC-AUC on booking-level cancellation predictions.
   * **Calibration:** Brier Score on booking-level probabilities (critical because $k$ depends directly on $\sum p_i$ and $\sqrt{\sum p_i(1-p_i)}$).
   * **Nightly Volume Accuracy:** Mean Absolute Error (MAE) between expected cancellations $\hat{\mu}_t$ and actual cancellations per hotel-night.

---

## 3. Chronological Validation Protocol

To mirror real-world deployment and prevent temporal leakage:
* **Train (`< 2016-07-01`):** $49,157$ bookings used to fit encoders, lookup tables, and ML classifiers.
* **Choose (`2016-07-01` to `2016-12-31`):** $29,546$ bookings ($368$ hotel-nights) used to tune the safety buffer ($z$ or $m$) and select the winning model.
* **Proof (`>= 2017-01-01`):** $40,687$ bookings ($486$ hotel-nights) locked for a single-shot out-of-sample evaluation.

---

## 4. Team Roles & Ownership

| Role | Primary Responsibilities | Key Deliverables |
|---|---|---|
| **Client Lead** | Business framing, error-bill economics, walk guardrail alignment, final presentation narrative. | `docs/CHARTER.md`, executive summary & ROI slides. |
| **Data Lead** | Raw data audit, temporal splitting, Bin 1/2/3 feature classification, Trap Experiment, AI benchmark audit. | `data/processed/hotels_model_ready.csv`, `notebooks/01_eda_and_cleaning.ipynb`, `docs/COLUMNS.md`, `docs/COMPARISON.md`. |
| **Model Lead** | Baseline & ML pipelines, drop-one feature validation, uncertainty buffer optimization, probability calibration. | `src/features.py`, `notebooks/Model.ipynb`, `docs/DECISIONS.md`. |
| **Build Lead** | Productionization of contract-compliant CLI tools, unit testing, artifact persistence, operational caveats. | `src/contract.py`, `src/backtest.py`, `src/nightly.py`, `models/best_overbook_model.joblib`, `docs/BUILD_PLAN.md`. |
