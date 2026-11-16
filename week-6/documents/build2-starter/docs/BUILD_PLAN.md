# Build Plan & Delivery Roadmap (`docs/BUILD_PLAN.md`)

## 1. Delivery Milestones (POC $\rightarrow$ MVP $\rightarrow$ Final)

| Stage | Deliverable | Command / Artifact | Status |
|---|---|---|---|
| **Contract & Tests** | Deterministic scoring engine & unit test suite | `pytest -q` (`src/contract.py`, `tests/test_contract.py`) | **Complete (5/5 tests pass)** |
| **POC (Backtest)** | Reproducible multi-policy benchmark on `Choose` & `Proof` | `python src/backtest.py` | **Complete** |
| **MVP (Inference CLI)** | Zero-retraining 14-night schedule generator from saved `.joblib` | `python src/nightly.py --from 2017-08-18 --nights 14` $\rightarrow$ `evidence/nightly_schedule.csv` | **Complete (28 hotel-nights)** |
| **Final Analysis** | Uncertainty vs. rooms buffer comparison, Trap Experiment, and capacity caveats | `notebooks/Model.ipynb`, `docs/COLUMNS.md`, `docs/BUILD_PLAN.md` | **Complete** |

---

## 2. Final Evaluation Scorecards (`src/backtest.py`)

### A. Choose Period (`2016 H2` — Selection & Buffer Tuning Window, 368 Hotel-Nights)

| Policy | Tuned Param | Choose AUC | Night MAE | Choose Cost (€) | Empty Rooms | Walked Guests | Walks / 1,000 | Ceiling Met |
|---|---|---|---|---|---|---|---|---|
| **1. Zero Oversell ($k=0$)** | `k=0` | — | 29.60 | €4,155,340.86 | 10,894 | 0 | 0.00 | True |
| **2. Client Flat Rule** (`floor(rate*book - m)`) | `m=16` | — | 9.19 | €2,414,409.40 | 5,532 | 169 | 9.06 | True |
| **3. Stratified Lookup Table** | `z=+1.80` | 0.7614 | 5.79 | €1,399,434.11 | 3,155 | 185 | 9.92 | True |
| **4. Logistic Regression (Bin-1)** | `z=+0.77` | 0.8174 | 6.14 | €1,175,613.16 | 2,713 | 186 | 9.97 | True |
| **5. GBDT (Bin-1, Uncertainty $z$)** | **`z=+0.22`** | **0.8440** | **6.40** | **€1,053,788.29** | **2,424** | **184** | **9.86** | **True** |
| **6. GBDT (Flat Rooms Buffer $m$)** | `m=1` | 0.8440 | 6.40 | €1,102,647.87 | 2,549 | 168 | 9.01 | True |

### B. Proof Period (`2017 Jan–Aug` — Single-Shot Out-of-Sample Evaluation, 486 Hotel-Nights)

| Policy | Param | Proof AUC | Brier | Night MAE | Proof Cost (€) | Empty Rooms | Walked Guests | Walks / 1,000 | Monthly Saving vs. Flat (€) |
|---|---|---|---|---|---|---|---|---|---|
| **1. Zero Oversell ($k=0$)** | `k=0` | — | — | 32.40 | €6,714,098.55 | 15,745 | 0 | 0.00 | -€347,275.37 |
| **2. Client Flat Rule** | `m=16` | — | — | 9.31 | €3,935,895.59 | 8,186 | 155 | 6.21 | €0.00 |
| **3. Stratified Lookup Table** | `z=+1.80` | 0.7694 | 0.1814 | 5.59 | €2,298,943.28 | 4,808 | 124 | 4.97 | +€204,619.04 |
| **4. Logistic Regression (Bin-1)** | `z=+0.77` | 0.8213 | 0.1706 | 6.89 | €2,146,343.55 | 4,346 | 108 | 4.33 | +€223,694.00 |
| **5. GBDT (Bin-1, Uncertainty $z$)** | **`z=+0.22`** | **0.8309** | **0.1684** | **7.44** | **€2,056,526.56** | **3,944** | **119** | **4.77** | **+€234,921.13** |
| **6. GBDT (Flat Rooms Buffer $m$)** | `m=1` | 0.8309 | 0.1684 | 7.44 | €2,126,485.42 | 4,106 | 110 | 4.41 | +€226,176.27 |

---

## 3. Uncertainty Buffer vs. Flat Rooms Buffer Analysis
Comparing Policy 5 ($k = \lfloor \hat{\mu} - 0.22 \cdot \hat{\sigma} \rfloor$) against Policy 6 ($k = \lfloor \hat{\mu} - 1 \rfloor$) on identical GBDT probabilities demonstrates the value of variance-aware buffering:
* **Why Uncertainty Wins:** On low-volume nights (e.g., Resort Hotel off-peak nights with $\hat{\sigma} \approx 1.8$), subtracting a flat $1$ room is overly conservative, whereas on high-volume peak nights ($\hat{\sigma} \approx 5.5$), subtracting $0.22 \cdot \hat{\sigma} \approx 1.2$ rooms scales risk dynamically with nightly uncertainty.
* **Net Gain:** The uncertainty buffer saves **€48,859.58** on `Choose` and **€69,958.86** on `Proof` over a flat room deduction while keeping `Proof` walks at **$4.77$ per $1,000$** (less than half the $10.0$ ceiling).

---

## 4. Production Deployment Caveats & Next-Phase Extensions

1. **Arrival-Cohort vs. Full Stay-Over Occupancy Caveat:**
   * `contract.nightly_table()` aggregates bookings by **`arrival_date`** (new check-ins arriving on night $t$). In live hotel inventory management, physical room capacity on night $t$ is shared between **new check-ins arriving on night $t$** and **multi-night guests staying over** from nights $t-1, t-2, \dots$.
   * **Operational Recommendation:** Integrate `oversell_k` from `src/nightly.py` as the **new-arrival overbooking allowance** on top of the PMS (Property Management System) stay-over house count, and cap total physical + virtual inventory at physical room capacity plus `oversell_k`.
2. **Dynamic Horizon Re-Scoring:**
   * Currently, all bookings are scored using strictly Bin-1 features fixed at creation. In Phase 2, separate horizon-specific models (e.g., $T-30\text{d}$, $T-7\text{d}$, $T-1\text{d}$) can safely incorporate verified Bin-2 updates (such as confirmed deposit payments or flight arrivals) at each decision checkpoint.
