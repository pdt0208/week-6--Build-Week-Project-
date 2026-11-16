# week-6--Build-Week-Project- 
Client Head- Nisha <br>
Build Head - Paddhati <br>
Data Head - Riya Hariya <br>
Model Head - Aditya Sharma 

# How many rooms can each hotel oversell tonight?

**Build 2, Week 6** · IIT Gandhinagar PG Diploma in AI-ML and Agentic AI Engineering

Our team is a revenue-science squad at RateGain. A two-hotel group in Portugal (a city hotel in Lisbon, a resort in the Algarve) asked us one question: **for every arrival night at each hotel, how many rooms beyond capacity can we keep selling, so that cancellations stop leaving rooms empty, without walking more than 10 guests per 1,000 arrivals?**

We answer it in euros, rooms and walks, on a period the model never saw.

> **The one-sentence answer.** Oversell each hotel-night by the number in the nightly table. Over January to August 2017 this rule would have cost **EUR 1.97 million** in empty rooms and walks instead of **EUR 3.88 million** under the client's flat rule (**49% less, about EUR 239,000 a month**), with **4.0 walks per 1,000 arrivals** against a ceiling of 10. The saving is real on nights near full (about 78% of it); on quiet nights it is an upper bound.

---

## Contents

1. [Results at a glance](#1-results-at-a-glance)
2. [Repository layout](#2-repository-layout)
3. [How to run it](#3-how-to-run-it)
4. [The three notebooks](#4-the-three-notebooks)
5. [How we cleaned the data, and why](#5-how-we-cleaned-the-data-and-why)
6. [Which columns we use, and which we drop](#6-which-columns-we-use-and-which-we-drop)
7. [The method: a ladder, from the client's rule to the best we can defend](#7-the-method-a-ladder-from-the-clients-rule-to-the-best-we-can-defend)
8. [What the data told us that the brief did not](#8-what-the-data-told-us-that-the-brief-did-not)
9. [Honest limits](#9-honest-limits)
10. [Reproducibility and the rules we followed](#10-reproducibility-and-the-rules-we-followed)
11. [Status, owners and sign-offs](#11-status-owners-and-sign-offs)
12. [AI use](#12-ai-use)
13. [Glossary](#13-glossary)
14. [Data, licence and setting](#14-data-licence-and-setting)

---

## 1. Results at a glance

### The proof: January to August 2017, run once

Every method was fitted on the train period, had its safety buffer chosen on the choose period, and was then scored by the client's own `src/contract.py` on the 486 proof nights.

| Method | AUC | Nightly error (rooms) | Empty rooms | Walks per 1,000 | Cost (EUR) | Saving a month vs flat rule (EUR) | Buffer |
|---|---|---|---|---|---|---|---|
| Client's flat rule | 0.55 | 9.3 | 8,050 | 6.70 | 3,882,930 | 0 | 15.5 rooms |
| Lookup table | 0.76 | 5.7 | 6,243 | 4.17 | 2,974,027 | 113,613 | 10 rooms |
| Logistic regression | 0.84 | 6.8 | 4,229 | 3.45 | 2,081,581 | 225,169 | 2 rooms |
| **Gradient boosting (chosen)** | 0.83 | 7.7 | **3,850** | 4.01 | **1,969,930** | **239,125** | 0 rooms |

- Every method respects the ceiling of 10 walks per 1,000.
- Selling nothing extra would have cost EUR 6.71 million.
- **AUC and money disagree at the top.** Logistic has the higher AUC (0.84 against 0.83) yet boosting is cheaper. AUC ranks single bookings; the client pays for the nightly total and for which side of it the error falls. We report euros.
- The logistic model with an uncertainty buffer comes within 1% of the winner (EUR 1.99M). The choice between the two is not strong; the selection rule (below) takes the cheaper.

### What the chosen rule does per hotel (proof period)

| Hotel | Saving vs flat rule | Walks per 1,000 | Nights near full (our proxy) |
|---|---|---|---|
| City Hotel | EUR 0.87M (54%) | 4.9 | 46% |
| Resort Hotel | EUR 1.04M (46%) | 2.5 | 57% |

### The trap experiment: why accuracy numbers lie

The same model trained three ways, read on the proof period:

| Run | Columns | Split | AUC |
|---|---|---|---|
| A | Bin 1 only | Time | **0.834** (the number the client can believe) |
| B | All three bins | Time | 1.000 (`reservation_status` is the answer itself) |
| C | Bin 1 only | Random | 0.879 (flattered by twin bookings and neighbouring weeks on both sides) |

---

## 2. Repository layout

```
.
├── README.md                     this file
├── BRIEF.md                      the client's brief (given)
├── AGENTS.md                     rules for coding agents (Build Lead writes)
├── requirements.txt              pinned versions (Python 3.12)
├── notebooks/
│   ├── 01_data_cleaning_eda.ipynb     cleaning, outliers, column bins, EDA
│   ├── 02_model.ipynb                 flat rule to boosting, buffers, proof
│   └── 03_claims.ipynb                the documents' questions as testable claims
├── src/
│   ├── data.py                   downloads and reads the bookings (given)
│   ├── contract.py               the client's scoring rule (given, never edited)
│   ├── features.py               our cleaning and the column bins
│   ├── backtest.py               TO WRITE: prints every method on proof, saves the model
│   └── nightly.py                TO WRITE: writes the 14-night table from the saved model
├── docs/                         CHARTER, RFP, PRD, BUILD_PLAN, COLUMNS, DECISIONS, ...
│   ├── decisions_log.csv         written by notebook 01
│   ├── column_bins.csv           written by notebook 01
│   ├── COLUMNS_table_for_paste.md
│   ├── DECISIONS_table_for_paste.md
│   └── claims_ledger.csv         written by notebook 03
├── evidence/                     charts and CSVs the slides cite
├── models/
│   ├── chosen.joblib             frozen model and buffer (written by notebook 02)
│   └── settings.json             every frozen setting
├── ai_logs/                      every AI chat, with INDEX.md
├── ai_benchmark/                 the AI's own build (on the ai-benchmark branch only)
└── data/                         hotels.csv, downloaded, never committed
```

### What each notebook writes

| Notebook | Files it writes |
|---|---|
| 01 | `evidence/eda_*.png`, `docs/decisions_log.csv`, `docs/column_bins.csv`, `docs/COLUMNS_table_for_paste.md`, `docs/DECISIONS_table_for_paste.md` |
| 02 | `models/chosen.joblib`, `models/settings.json`, `evidence/model_*.png`, `evidence/method_table_choose.csv`, `evidence/method_table_proof.csv`, `evidence/error_bill_proof.csv`, `evidence/trap_auc_proof.csv`, `evidence/choose_probs.csv`, `evidence/choose_nights.csv`, `evidence/proof_nights.csv`, `evidence/proof_probs.csv` |
| 03 | `evidence/claims_*.png`, `docs/claims_ledger.csv` |

---

## 3. How to run it

### On Google Colab (nothing to install)

1. Upload the three notebooks, or open them from GitHub with
   `https://colab.research.google.com/github/YOUR_USER/YOUR_REPO/blob/main/notebooks/01_data_cleaning_eda.ipynb`
2. Run **notebook 1, then 2, then 3, in the same session** (Runtime, then Run all).
3. Each notebook starts with a setup cell that creates the folders and writes `contract.py`, `data.py` and `features.py` into `src/` if they are missing. The first run downloads the bookings (needs internet).
4. Notebook 2 takes a few minutes (it fits about 30 models for the drop-one test).

Notebook 3 reads files that notebook 2 saves. If you run them in separate Colab sessions, the files are gone; run 1 and 2 again first.

### Locally or in a Codespace

```bash
pip install -r requirements.txt
python src/data.py                 # downloads data/hotels.csv, prints 119,390 bookings, 32 columns
jupyter lab notebooks/             # then run 01, 02, 03 in order
```

`pip install` uses the pinned versions in `requirements.txt` (pandas 3.0.6, scikit-learn 1.9.1, and so on). The notebooks were built and tested with exactly those. On Colab's own, usually older, libraries they should also run, but the numbers could differ in the last digit.

### The two commands the project must have by Friday

```bash
python src/backtest.py                              # every method on the proof period; saves the chosen one to models/
python src/nightly.py --from 2017-08-18 --nights 14 # the table the revenue manager reads each morning
pytest -q                                            # checks the client's scoring rule
```

`backtest.py` and `nightly.py` are **not written yet** (see [section 11](#11-status-owners-and-sign-offs)). The logic they need already exists in notebook 02 and `models/chosen.joblib`.

---

## 4. The three notebooks

### Notebook 1: `01_data_cleaning_eda.ipynb` (Data Lead)

Loads the raw bookings and cleans them one decision at a time. Each step has the same five lines: **Problem, Evidence, Options, Decision, Why not the others.** It then checks outliers column by column, bins all 32 columns, and draws the EDA charts. It ends by proving that its hand-made cleaning equals `features.clean()`.

### Notebook 2: `02_model.ipynb` (Model Lead)

Climbs a ladder, and only keeps climbing while each rung beats the one below on the choose period. The proof period stays locked behind the switch `RUN_PROOF` until every setting is frozen. A dry run first applies the exact proof code to the *choose* period and must reproduce the choose table to the cent; only then is the proof run, once.

### Notebook 3: `03_claims.ipynb` (all owners)

Turns every statement the documents ask about into a **claim, test, witness and verdict**, with the verdict computed from the data. It covers the brief's numbers, the three observations, the wish list, the full-night caveat, the travel-partner view and the panel's likely questions, and ends with a claims ledger and draft answers for the charter, PRD and slide 1.

---

## 5. How we cleaned the data, and why

**One rule behind every choice: no row is removed.** The client scores a night by counting the bookings due and the bookings that cancelled, so deleting a row changes the client's own numbers. We fix values, add flags, or group rare labels. All 119,390 rows survive.

| Step | Decision | Reason, with the number |
|---|---|---|
| D1 | Never remove a row | Every booking is a room the hotel handled that night |
| D2 | Fill 4 empty `children` with 0 | 92.8% of bookings have no children; 0 is the median |
| D3 | Fill 488 empty `country` with `UNK` | They cancel at 13.7% against 37.0% overall, so "unknown" is its own kind of booking; filling with PRT would invent a fact |
| D4 | Empty `agent` becomes `None`; `company` ID becomes a yes/no flag | Empty means "booked without one" (DOMAIN.md); none-agent cancels at 24.7% vs 39.0%; 94.3% of company is empty |
| D5 | Merge meal `Undefined` into `SC` | Same meaning, two spellings |
| D6 | **Keep all 31,994 duplicate rows** | No booking ID; twins are 41.6% Groups and cancel at 58% vs 26%, so they look like multi-room bookings. Deleting would remove **22.7% of the choose-period bookings and 39.2% of its cancellations** and pull the cancel rate from 36.9% to 29.0% |
| D7 | Keep 715 zero-night bookings, add flag | They occupy a room; adr is 0 so they are worth EUR 0 |
| D8 | Keep 180 no-guest bookings, add flag | Look like blocked rooms |
| D9 | Cap `adults` at 4, `children` at 3, `babies` at 2; keep raw `adr` for scoring, add clipped `adr_clean` | 13 rows with 10 to 55 adults, all cancelled in one resort in Sept to Oct 2015: placeholders. The odd `adr` rows move no scored night by more than a euro |
| D10 | Group labels with fewer than 300 **train** bookings into `Other` | Sample-size rule; counted on train only so nothing leaks |
| D11 | Use month-of-arrival and weekday; do not use year, week number or day of month | Proof is all 2017, a year the model never saw |

### Outliers: keep, cap or flag

The usual IQR rule misfires on skewed, mostly-zero columns (it flags thousands of normal rows in `children`), so we looked at the extreme percentiles, the IQR count, whether extreme rows *behave differently*, and read the worst rows one by one.

- **Kept, because they are real:** `lead_time` up to 737 days (147 bookings above 600 days, 98.6% cancelled, almost all Groups: long-ahead group contracts); long stays up to 69 nights; guests with up to 26 previous cancellations.
- **Capped, because they are slips or placeholders:** the head-count columns above.
- **Left raw for scoring:** `adr`, because the client's money rule uses it as it is and `contract.py` must not change. A EUR 5,400 booking sits in the train period (never scored); a EUR -6.38 booking moves its night by under one euro.

---

## 6. Which columns we use, and which we drop

Every one of the 32 columns sits in exactly one bin, by **when the hotel learns its value**.

| Bin | Meaning | Count | Used by the model? |
|---|---|---|---|
| 1 | Fixed at booking | 18 | Yes: 12 columns after pruning |
| 2 | Amendable before arrival | 8 | Only tried, with a caveat |
| 3 | After the fact | 5 plus the outcome | Never |

### The final model reads these 12 columns

`hotel`, `lead_time`, `country_grp`, `segment_grp` (market segment), `agent_grp`, `deposit_type`, `customer_type`, `room_grp` (reserved room type), `previous_cancellations`, `previous_bookings_not_canceled`, `is_repeated_guest`, `has_company`.

### Dropped, and why

| Group | Columns | Reason |
|---|---|---|
| Leaks (bin 3) | `reservation_status`, `reservation_status_date`, `assigned_room_type`, `booking_changes`, `required_car_parking_spaces` | Written at check-in or when the outcome is known. Parking is 0 for every cancelled booking while about 10% of arrivals have one; `reservation_status` **is** the answer |
| Unstable | `arrival_month`, `arrival_dow`, `meal` | A drop-one test showed the model got *better* without them. Removing `arrival_month` lifted boosting's AUC by **0.061**: the seasonal shape of 2015 did not repeat in 2016 (City July cancelled 67% in 2015, 33% in 2016) |
| Redundant or useless | `arrival_date_year`, `arrival_date_week_number`, `arrival_date_day_of_month`, `distribution_channel`, raw `company` ID | Year is never seen in training; week and day repeat the date; the channel adds 0.0004 AUC once the segment is in |

### Tried with a caveat (bin 2), not used

Stay length, adults, children, babies, waiting-list days, `adr` and `total_of_special_requests`. All but one add 0.002 AUC or less. **Special requests adds 0.043**, which is the warning sign: it holds the count *at the end* (check-in or cancellation), so it would let the model peek at later information. The project rule says only bin 1 feeds the MVP, and we keep to it.

The full table, with the evidence for each column, is in `docs/column_bins.csv` and notebook 1, section 4.

---

## 7. The method: a ladder, from the client's rule to the best we can defend

Every method does the same two steps: (1) give each booking a **probability of cancelling**, (2) add the probabilities per hotel and night into **expected cancellations**, then oversell `k = floor(expected - buffer)` rooms, never below 0. The buffer is a safety margin chosen once on the choose period.

| Version | Method | Choose cost (EUR) |
|---|---|---|
| V0 | Sell nothing extra | 4,155,341 |
| V1 | **The client's flat rule**: hotel rate times bookings, less a fixed number of rooms (15.5 needed to keep the ceiling) | 2,387,930 |
| V2 | Lookup table by hotel, deposit type, segment and lead-time band, shrunk toward the hotel rate when a group is small | 1,885,618 |
| V3 | Logistic regression on the 12 columns | 1,110,823 |
| V4 | **Gradient boosting**, depth 2, 150 rounds, min leaf 200, L2 = 1 | **987,194** |

The rule for picking the winner was written before looking: **lowest choose-period cost with the ceiling met; if a simpler method is within 1% of the best, take the simpler one.** The same style of rule picked the boosting settings (take the best AUC; if a simpler setting is within 0.001, take the simpler) and the columns to drop (drop a column if removing it raises AUC by 0.005 for one model and costs the other less than 0.002).

### What went wrong on the way, and what we learned

1. **The flexible model was worse than the simple one.** With all 15 candidate columns, boosting scored AUC 0.778 against logistic's 0.825, and both under-predicted cancellations.
2. **We looked at where they were wrong, not just how wrong.** The error sat in Online TA bookings without a deposit: they grew from 36% to 54% of all bookings between train and choose, and their cancel rate rose from about 30% to 38%. That is **drift**, not a bug.
3. **The drop-one test found the culprit:** `arrival_month`. Removing three unstable columns turned boosting from the worst learned model (cost EUR 1.68M) into the best (EUR 0.99M). **Column choice mattered more than algorithm choice.**
4. **The buffer absorbs drift** (a negative buffer means "oversell more than the model expects"), but a refitted model needs its buffer chosen again.

### The buffer: rooms or uncertainty?

`spread` is the standard deviation of a night's total *if bookings cancelled independently*. Real cancellations are correlated (a group cancels together) and models make mistakes, so the real nightly error (8.6 rooms for the chosen model) is about **three times** the claimed spread (3.0 rooms). An uncertainty buffer (`expected - z x spread`) helps weak methods, for example it cuts the flat rule from EUR 2.39M to EUR 2.16M on choose because the margin grows with the size of the night, but for the final model the best multiplier is 0, so the simpler rooms buffer wins.

### What would flip the choice?

Nothing in the ceiling. On the choose period our method is 54% to 60% cheaper than the flat rule at ceilings of 5, 10 and 20 walks per 1,000. The ceiling sets the price of safety: relaxing it from 10 to 20 cuts cost by about 19%, tightening it to 5 raises it by about 25%.

If a walk cost **EUR 1,000 instead of 300**, nothing changes under the client's ceiling (the best buffer stays 0, because the ceiling is already the binding limit). Without a ceiling the best buffer would move from -5.5 to -1.5 rooms and walks would fall from 52 to 16 per 1,000.

---

## 8. What the data told us that the brief did not

Every statement in the brief is a claim with a test (see `docs/claims_ledger.csv`, 35 rows).

### Numbers in the brief

| Brief says | Data says | Verdict |
|---|---|---|
| About one booking in four is cancelled | **37.0%** (City 41.7%, Resort 27.8%) | False |
| The flat rule's "a few rooms for safety" is enough | 3 rooms walk **55 per 1,000**; it needs 15.5 rooms | False |
| 10 walks per 1,000 is "one guest in a hundred" | 1.0%, on a base of arrivals | True |
| 119,390 bookings, 32 columns, July 2015 to August 2017 | Matches | True |
| A walk costs EUR 300; an empty room costs adr x nights | Mean booked stay City EUR 319, Resort EUR 435 | Cannot check the EUR 300 |

### Observation 1: "Most prepaid, non-refundable bookings never turn into a stay. That cannot be right."

**Finance is right.** Non-refundable bookings cancel **99.4%** of the time (14,587 bookings): City 99.8%, Resort 96.0%, and 99% or more in each year. They are 12% of bookings but **33% of all cancellations**, and come overwhelmingly from Portugal (97%) through Groups (63%) and offline agents, at the City Hotel (88%). The data cannot say *why*, so we send the client one question: **"When a non-refundable booking is cancelled or does not show, does the hotel keep the prepayment, and is `adr` the rate charged or only the rate quoted?"** (It changes what an empty room really costs.)

A subtle point: removing `deposit_type` from the model costs only 0.005 AUC, not because it is unimportant but because `country`, `agent` and `lead_time` identify the same bookings. The information is shared.

### Observation 2: "A vendor's daily list of likely cancellations was dropped by the front desk within a month."

The list is the wrong product. At a 0.5 cut it has about 22 names a night at the City Hotel, 18% of the names would have turned up anyway, and 61% of real cancellations are missing from it. Counting the names also estimates the night's cancellations worse than adding up the probabilities (off by 15.5 rooms a night against 6.4). **So the MVP outputs a number of rooms per hotel and night with a one-line reason, not a list of guests.**

### Observation 3: "The vendor's demo showed 95% accuracy. In the lobby it never came close."

On this data **only one variant reaches 95%: the one that includes `reservation_status`**, the answer itself. Adding every other late column reaches about 80%; a random split adds 3 points to the honest time split. An honest model scores **74% accuracy (63% for "always predict stays") and AUC 0.84**. We cannot say exactly what the vendor did, but this is what the desk gets with only bin-1 columns.

### The wish list: "hidden guest types"

That is clustering. We tested it: k-means on the bin-1 columns (K = 6, weakly separated), added to the model as a feature. AUC moves by **-0.0003** and the euro cost by **+0.8%**. The most striking cluster (5,007 bookings, 100% cancelled, median lead time 247 days, mostly Groups) is the long-lead group business that `lead_time`, `segment` and `deposit_type` already capture. **A worthwhile marketing project, outside this MVP.**

### The travel-partner view

52% of the 218 travel agents have under 30 train bookings (together only 2.8% of agent volume), and 61% of those show a cancel rate of exactly 0% or 100%, against 2% for agents with 100 or more: noise, not behaviour. Hence the "fewer than 300 train bookings becomes `Other`" rule.

---

## 9. Honest limits

- **The full-night caveat.** The client's money rule treats every cancellation as an empty room worth `adr x nights`, which is true only on nights the hotel would have been full. Capacity is not in the data, so we built a **proxy**: rooms in house per night, capacity = the 99th percentile, "near full" = at least 90% of it. By that proxy only **46% of City nights and 57% of Resort nights** are near full, and **78% of the saving (EUR 1.50M of EUR 1.91M)** falls on near-full nights. **The honest range is EUR 1.5M to EUR 1.9M.** At the resort in January, rooms in house average 55% of the proxy and no night is near full, so the oversell number means nothing there. Change the 90% and the shares move; the shape does not.
- **Walks came out at 4.0 per 1,000, well under the 10 allowed.** Cancellations kept drifting upward (train 35.8%, choose 36.9%, proof 38.7%), so a buffer chosen on choose was a little cautious and money was left on the table. We did **not** re-tune it, because that would mean choosing on the proof period.
- **We did not refit on train plus choose.** The contract allows it, but the buffer chosen for a train-only model would not fit a refitted one. A rolling refit with a rolling buffer is on the roadmap.
- **The nightly table treats every booking as already on the book** (the contract scores each night's final book). About half of the final book is there 14 days ahead, so a run using only the bookings known that morning is the natural next step. The typical cancellation happens 54 days before arrival and only 14% in the last week, which is encouraging.
- **One year of seasonality is not enough.** Month-of-year hurt the model, so the MVP has no seasonal feature. A second full year in train would let it be tested again.
- **The data is a public teaching copy** (TidyTuesday, 11 February 2020) of two real Portuguese hotels. The walk cost, walk ceiling and the brief are this course's construction.
- **Duplicate rows** might be double entries; we judged them to be multi-room bookings from the evidence, and would ask the client.

---

## 10. Reproducibility and the rules we followed

- **Time split everywhere.** Train: arrivals before 1 July 2016 (49,157 bookings). Choose: 1 July to 31 December 2016 (29,546). Proof: 1 January to 31 August 2017 (40,687), run once.
- **The proof was locked** behind `RUN_PROOF` until every setting was frozen. Before that, a dry run applied the proof code to the choose period and reproduced the choose table exactly. After the proof, nothing was changed.
- **`contract.py` is never edited.** Every method is scored by it, on the same nights. Notebook 3's "walk cost EUR 1,000" test temporarily sets `contract.WALK_COST` and asserts it is restored.
- **Every saving is reported against the client's flat rule**, formalised as the client described it (rate times bookings, less a fixed number of rooms), never against selling nothing extra.
- **Results are deterministic.** All random seeds are fixed (`random_state=0`). A rebuild from an empty folder gave identical proof numbers (EUR 1,969,929.79).
- **Euros, rooms and walks per 1,000 are always reported together.**

---

## 11. Status, owners and sign-offs

### Roles

| Role | Name | Owns |
|---|---|---|
| Client Lead | | `docs/CHARTER.md`, `docs/RFP.md`, the travel-partner view, the full-night caveat, slides 1, 2, 3, 10 |
| Data Lead | | profiling notebook, `docs/DECISIONS.md`, `docs/COLUMNS.md`, `src/features.py`, the AI benchmark, `docs/COMPARISON.md`, `ai_logs/INDEX.md`, slides 4, 9 |
| Model Lead | | `src/backtest.py`, the method table, the buffer, slides 5, 6 |
| Build Lead | | `docs/PRD.md` with the diagram, `docs/BUILD_PLAN.md`, `src/nightly.py`, `AGENTS.md`, `docs/ROADMAP.md`, slides 7, 8 |

### Who owns each artifact, and who has signed it

A challenger signs by writing the commit hash they checked. An artifact counts as done only once it is signed.

| Artifact | Owner | Challenger | Signed at commit |
|---|---|---|---|
| `docs/CHARTER.md` and `docs/RFP.md` | | | |
| `docs/COLUMNS.md` and `src/features.py` | | | |
| `src/backtest.py` | | | |
| `docs/PRD.md`, `docs/BUILD_PLAN.md` and `src/nightly.py` | | | |
| `docs/COMPARISON.md` | | | |

### What is done and what is left

| Area | State |
|---|---|
| Notebooks 1 to 3, `src/features.py`, evidence charts and CSVs | Done |
| 10-slide deck | Slides 1 to 7 and 10 done; slides 8 and 9 are frames to fill from the grill log and the AI benchmark |
| Column table and decisions log for the docs | Ready to paste |
| `src/backtest.py`, `src/nightly.py`, `pytest -q` | Not written / not run |
| Charter v0 on paper, RFP, PRD with architecture diagram, BUILD_PLAN | Not written |
| `AGENTS.md`, both grill runs, AI benchmark, `docs/COMPARISON.md`, `ai_logs/INDEX.md` | Not done |
| Evidence screenshots in `evidence/`, README sign-offs, recordings | Not done |

### Where are our recordings?

| Member | Role | Recording link on the LMS |
|---|---|---|
| | Client Lead | |
| | Data Lead | |
| | Model Lead | |
| | Build Lead | |

---

## 12. AI use

Per `docs/AI_WORKFLOW.md`: everything except Tuesday's paper frame and Saturday's defence may use AI, **every chat is logged** in `ai_logs/` (one file per session, named `YYYYMMDD_role_tool_purpose.md`, indexed in `ai_logs/INDEX.md`), and **whoever commits an AI-written line must be able to explain it.**

The three notebooks and `src/features.py` were produced with Claude's help. Every member should read them end to end before Saturday, because the panel picks who answers and asks about teammates' numbers too. The notebooks are written so that each step carries its own reason; if you cannot explain a line, that line is a risk.

---

## 13. Glossary

| Term | Meaning here |
|---|---|
| **Oversell `k`** | Rooms sold beyond capacity for one hotel on one night |
| **Walk** | A guest with a booking who arrives to a full hotel and is sent elsewhere at the hotel's cost (EUR 300) |
| **Ceiling** | At most 10 walks per 1,000 arrivals; a rule that breaks it is rejected |
| **Arrivals** | Scored bookings that did not cancel |
| **Stay value `V`** | A night's mean of `adr` x nights; the cost of an empty room |
| **Buffer** | A safety margin subtracted from expected cancellations; chosen once on the choose period, then frozen |
| **AUC** | How well probabilities rank cancellers above stayers (0.5 = chance, 1.0 = perfect) |
| **Bin 1, 2, 3** | Columns fixed at booking, amendable before arrival, or written after the outcome |
| **Drift** | The booking mix and cancel rates change over time, so a model trained on the past is always a little behind |
| **Near full** | Rooms in house at least 90% of a capacity proxy (the 99th percentile of rooms in house) |

---

## 14. Data, licence and setting

- **Source:** Antonio, N., de Almeida, A. and Nunes, L. (2019). Hotel booking demand datasets. *Data in Brief*, 22, 41 to 49. https://doi.org/10.1016/j.dib.2018.11.126
- **Licence:** CC BY 4.0, so every copy of our work cites the source above.
- **Copy used:** the TidyTuesday version of 11 February 2020, 119,390 bookings and 32 columns, downloaded by `python src/data.py`. The file is never committed (`.gitignore` keeps `/data/` out).
- **Setting:** RateGain Travel Technologies is a real company, but it supplied neither this brief nor this data. The walk cost, the walk ceiling and the brief itself are this course's construction for teaching.
