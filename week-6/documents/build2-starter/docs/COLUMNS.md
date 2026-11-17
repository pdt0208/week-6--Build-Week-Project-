# Dataset Columns & Feature Binning (`docs/COLUMNS.md`)

To prevent target and post-booking data leakage, every column in the raw `hotels.csv` dataset ($119,390 \times 32$) is classified into one of three temporal bins based on **when the information becomes fixed and known**:

* **Bin 1 (Fixed at Booking):** Known with certainty on the day a reservation is created. **Only Bin-1 features are permitted in production models.**
* **Bin 2 (Amendable Before Arrival):** Can be modified by the guest or revenue management after booking (e.g., length of stay, guest counts, negotiated ADR, special requests). Excluded from production modeling.
* **Bin 3 (Post-Arrival / Outcome):** Known only at check-in, cancellation, or check-out. Strictly excluded (direct target leakage).

---

## 1. Complete 32-Column Classification Table

| # | Column Name | Bin | Production Status | Rationale / Notes |
|---|---|---|---|---|
| 1 | `hotel` | **Bin 1** | **Included** | Property identifier (`City Hotel` vs. `Resort Hotel`), fixed at booking. |
| 2 | `is_canceled` | **Target** | **Target ($y$)** | Binary outcome ($1$ = canceled, $0$ = checked in). |
| 3 | `lead_time` | **Bin 1** | **Included** | Days between booking creation and scheduled arrival date. |
| 4 | `arrival_date_year` | **Bin 1** | Excluded | Excluded to prevent temporal overfitting (`Train` is 2015–2016; `Proof` is all 2017). |
| 5 | `arrival_date_month` | **Bin 1** | Excluded | Dropped after drop-one check showed 2015 seasonal spikes hurt `Choose` AUC by $-0.0687$. |
| 6 | `arrival_date_week_number` | **Bin 1** | Excluded | Dropped alongside `arrival_date_month` to prevent calendar-year memorization. |
| 7 | `arrival_date_day_of_month` | **Bin 1** | **Included** | Captures intra-month booking/arrival patterns without year-specific overfitting. |
| 8 | `stays_in_weekend_nights` | **Bin 2** | Excluded | Guests can shorten or extend stays prior to arrival. |
| 9 | `stays_in_week_nights` | **Bin 2** | Excluded | Guests can shorten or extend stays prior to arrival. |
| 10 | `adults` | **Bin 2** | Excluded | Party size can be amended before check-in. |
| 11 | `children` | **Bin 2** | Excluded | Party size can be amended before check-in. |
| 12 | `babies` | **Bin 2** | Excluded | Party size can be amended before check-in. |
| 13 | `meal` | **Bin 1** | **Included** | Meal package selected at booking (`BB`, `FB`, `HB`, `SC`, `Undefined`). |
| 14 | `country` | **Bin 1** | **Included** | Guest origin country; levels with $<300$ train bookings grouped into `"Other"`. |
| 15 | `market_segment` | **Bin 1** | **Included** | Booking channel segment (`Online TA`, `Offline TA/TO`, `Groups`, `Direct`, `Corporate`, etc.). |
| 16 | `distribution_channel` | **Bin 1** | **Included** | Booking distribution channel (`TA/TO`, `Direct`, `Corporate`, `GDS`, `Undefined`). |
| 17 | `is_repeated_guest` | **Bin 1** | **Included** | Historical loyalty flag known when the booking is placed. |
| 18 | `previous_cancellations` | **Bin 1** | **Included** | Count of prior canceled bookings by the guest profile at booking time. |
| 19 | `previous_bookings_not_canceled` | **Bin 1** | **Included** | Count of prior honored stays by the guest profile at booking time. |
| 20 | `reserved_room_type` | **Bin 1** | **Included** | Room category originally booked by the guest. |
| 21 | `assigned_room_type` | **Bin 3** | Excluded | Frequently updated at check-in or during pre-arrival room allocation. |
| 22 | `booking_changes` | **Bin 2** | Excluded | Accumulates post-booking modifications over the lead-time window. |
| 23 | `deposit_type` | **Bin 1** | **Included** | Deposit policy (`No Deposit`, `Non Refund`, `Refundable`) set at booking. |
| 24 | `agent` | **Bin 1** | **Included** | Travel agency ID (`"None"` preserved); IDs with $<300$ train rows grouped into `"Other"`. |
| 25 | `company` | **Bin 1** | **Transformed** | Binarized into `has_company` (`"1"` if `company != "None"`, else `"0"`). |
| 26 | `days_in_waiting_list` | **Bin 2** | Excluded | Accumulates while a reservation sits on a waitlist post-creation. |
| 27 | `customer_type` | **Bin 1** | **Included** | Contract classification (`Transient`, `Transient-Party`, `Contract`, `Group`). |
| 28 | `adr` | **Bin 2** | Excluded | Average Daily Rate can be renegotiated/discounted before arrival; used only in `contract.py` scoring. |
| 29 | `required_car_parking_spaces` | **Bin 2** | Excluded | Often requested or modified after initial booking. |
| 30 | `total_of_special_requests` | **Bin 2** | Excluded | Special requests accumulate up to the arrival date. |
| 31 | `reservation_status` | **Bin 3** | Excluded | Direct target leakage (`Canceled`, `Check-Out`, `No-Show`). |
| 32 | `reservation_status_date` | **Bin 3** | Excluded | Timestamp of final status change (post-outcome). |

---

## 2. The Trap Experiment (Leakage & Split Ablation)

To verify why strict temporal splitting and Bin-1 feature discipline are mandatory, we evaluated the exact same `HistGradientBoostingClassifier` pipeline across three experimental setups:

| Run | Feature Set | Split Strategy | Proof / Test AUC | Operational Interpretation |
|---|---|---|---|---|
| **Run A (Trustworthy Baseline)** | **First bin only (15 Bin-1 features)** | **Temporal Split** (`Train < 2016-07-01`, `Proof >= 2017-01-01`) | **0.8299** | True day-of-booking discrimination achievable in live hotel operations. |
| **Run B (Post-Booking Leakage)** | **All three bins (Bin 1 + Bin 2 + Bin 3)** | **Temporal Split** | **1.0000** | **"95%+ accuracy in the lobby" trap:** `reservation_status` and post-booking edits reveal the answer after the fact. |
| **Run C (Future-to-Past Leakage)** | **First bin only (15 Bin-1 features)** | **Random Stratified Split** | **0.9129** | Artificially inflated by $+0.0830$ AUC because future bookings from the same group/week leak into training. |
