# Engineering & Modeling Decisions Log (`docs/DECISIONS.md`)

This document records the key architectural, data engineering, and optimization decisions made throughout the project, along with empirical justifications.

---

### Decision 1: Preserve Literal `"None"` Strings in `agent` and `company`
* **Context:** In `hotels.csv`, missing `agent` and `company` values are encoded as the literal string `"None"`, which `pandas.read_csv()` parses as `NaN` by default.
* **Decision:** Load all datasets with `keep_default_na=False` so `"None"` (direct/non-agency bookings) remains an explicit, informative category rather than missing data.
* **Impact:** Eliminates missing-value imputation artifacts across $16,340$ non-agent bookings and $112,593$ non-corporate bookings.

### Decision 2: Retain Zero-Night / Zero-ADR Rows in Scoring (`nights == 0`)
* **Context:** Inspection of `hotels_model_ready.csv` revealed rows where `nights = 0` and `adr = 0.0` (e.g., day-use bookings or walk-in billing adjustments).
* **Decision:** Retain all $119,390$ rows without filtering so `contract.nightly_table()` and `contract.night_cost()` score against the exact unaltered client dataset.

### Decision 3: Drop `arrival_date_year`, `arrival_date_month`, and `arrival_date_week_number`
* **Context:** Because `Train` ends on `2016-06-30` and `Proof` spans `2017-01-01` to `2017-08-31`, year-level features cannot generalize, and calendar month/week features risk memorizing 2015 seasonal anomalies.
* **Decision:** Conduct a drop-one validation check on `Choose` (`2016 H2`). Removing `arrival_date_month` and `arrival_date_week_number` improved `Choose` AUC from **$0.7753$ to $0.8440$ ($+0.0687$ AUC)** while keeping `arrival_date_day_of_month` and `lead_time`.

### Decision 4: Group High-Cardinality Categories (`< 300` Train Bookings $\rightarrow$ `"Other"`) & Binarize `company`
* **Context:** `agent` ($219$ unique IDs) and `country` ($177$ unique codes) exhibit heavy tail drift—$69$ unseen agent IDs and $28$ unseen countries appear in `Proof`. Passing raw integer codes to trees creates arbitrary numeric splits.
* **Decision:** Implement `RareCategoryGrouper(min_count=300)` inside `src/features.py` (fit strictly on `Train`), mapping rare levels to `"Other"` and converting `company` into a binary `has_company` flag. Pass categorical indices natively to `HistGradientBoostingClassifier`.

### Decision 5: Conservative Floor Rounding & Extended Buffer Grid ($z \in [-1.5, 2.5]$)
* **Context:** Using `np.round()` adds $+0.5$ rooms of oversell bias on average per night, and restricting $z \ge 0$ prevents discovering if calibrated probabilities allow slightly more aggressive overbooking under the walk guardrail.
* **Decision:** Define the uncertainty policy as $k = \max\left(0, \lfloor \hat{\mu} - z \cdot \hat{\sigma} \rfloor\right)$ where $\hat{\mu} = \sum_i p_i$ and $\hat{\sigma} = \sqrt{\sum_i p_i(1-p_i)}$, searching $z \in [-1.5, 2.5]$ in steps of $0.01$ strictly on `Choose` subject to `walks_per_1000 <= 10.0`.

### Decision 6: Programmatic Winner Selection Strictly on `Choose`
* **Context:** Selecting a model after viewing `Proof` results introduces test-set leakage.
* **Decision:** Enforce an automated selection rule (`min(choose_cost)` subject to `choose_ceiling_met == True`). **GBDT (Bin-1, Uncertainty $z=+0.22$)** won on `Choose` (€$1,053,788.29$ at $9.86$ walks per $1,000$) and was frozen before evaluating `Proof` (€$2,056,526.56$ at $4.77$ walks per $1,000$).
