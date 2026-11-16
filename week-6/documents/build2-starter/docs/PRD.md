# What exactly are we building, and why this way?

**Owner:** Build Lead. **Closes:** ladder stages 3 and 5. **Length:** two pages and one diagram.

Write the options section on Wednesday, before the method table exists, and update it once the table is in. The architecture goes in on Thursday.

### 1. Who uses it, when, and what do they do with it?


**Our answer:**

- The primary user is the hotel revenue manager.
- The manager uses the system every morning to review the next 14 arrival nights.
- The system provides the number of bookings, expected cancellations, safety buffer and recommended number of extra rooms to sell.

### 2. Which two ways could we decide, what does each cost, which do we choose, and what would flip it?


**Our answer:**Option A: the client's flat rule. Every booking at a hotel gets that hotel's average cancellation rate from the training period. Costs nothing to run and is easy to explain. On the proof period (2017 Jan–Aug) it costs EUR 3.42M: 7,188 empty rooms and 185 walked guests (7.42 per 1,000).
Option B: a per-booking cancellation model. A gradient-boosted tree gives each booking its own cancellation probability from the 26 columns known at booking time. It costs training, a refit schedule and monitoring. On the same proof period it costs EUR 1.63M: 3,023 empty rooms and 194 walked guests (7.78 per 1,000).
We choose B. It cuts the error bill by about 53% and stays under the 10-per-1,000 walk ceiling. It also beat the other three models we tried: decision tree EUR 1.95M, logistic regression EUR 2.29M, random forest EUR 1.67M. Selling nothing extra (k = 0) would cost EUR 6.71M.
What would flip it: (a) the saving shrinks below the cost of running and monitoring the model on a fresh period; (b) the walk rate on new data goes above 10 per 1,000; (c) the client needs a rule a manager can recompute by hand. If only (c) matters, fall back to A with a per-segment rate.
- 

### 3. What is our answer to the wish-list item?


**Our answer:**TO FILL — the wish-list item is not in either notebook. Paste its wording and I will write the paragraph. Template: "The item asks for [X]. It needs [flat rule / per-booking model / something else]. Its output would [not change / change] the extra-rooms number the revenue manager acts on, because [reason]. Effect on proof-period cost: [EUR or 'none']."

### 4. What exactly is in the table the client reads?

**Our answer:**One row per hotel and arrival night for the next 14 nights: hotel, arrival date, rooms booked, expected cancellations, buffer, extra rooms to sell, and a one-line reason.

### 5. Which inputs do we use, by bin?


**Our answer:**Numeric (15): lead_time, arrival_date_year, arrival_date_day_of_month, arrival_date_week_number, stays_in_weekend_nights, stays_in_week_nights, nights, adults, children, babies, total_guests, is_repeated_guest, previous_cancellations, previous_bookings_not_canceled, adr.
Categorical (11): hotel, arrival_date_month, meal, country, market_segment, distribution_channel, reserved_room_type, deposit_type, agent, company, customer_type.
Used for splitting, grouping and scoring only, not as model inputs: arrival_date, is_canceled (target), and adr × nights (stay value).
Dropped in cleaning: Bin 2 (can change after booking): booking_changes, days_in_waiting_list, required_car_parking_spaces, total_of_special_requests. Bin 3 (leakage): reservation_status, reservation_status_date, reservation_year, assigned_room_type.

### 6. Where do the numbers come from, and how is the buffer set and frozen?

**Our answer:**Each booking gets a cancellation probability from the gradient-boosted model, trained on bookings arriving before 2016-07-01 (49,157 bookings). Probabilities are summed per hotel and arrival night to give expected cancellations; the spread is √Σp(1−p).

Extra rooms to sell = max(0, round(expected − z × spread)). The buffer multiplier z is searched from 0 to 3 in 301 steps on the choose period (2016-07-01 to 2016-12-31, 29,546 bookings). We keep the cheapest z that holds walks at or below 10 per 1,000 arrivals, with walks costed at EUR 300 and empty rooms at the night's average stay value (adr × nights).

The chosen z = 0.0 is then frozen and applied unchanged to the proof period (2017-01-01 onward, 40,687 bookings, 486 hotel-nights).

### 7. What is the error bill on the proof period?

Euros, rooms and walks per 1,000 for the chosen rule and for the client's flat rule, from python src/backtest.py.

Saving = (3,423,182 − 1,625,177) ÷ 8 months (Jan–Aug 2017). Figures are from the notebook run; re-run python src/backtest.py to confirm they match.

### 8. What does the system look like, first as a black box and then inside?

_Two drawings. The black box: what goes in, what comes out, who reads it. The white box: every step from the bookings file to the table, with the file or function that does it. Then list what production would add._

**Our diagram:** `docs/architecture.png`
Black box: in = the hotel bookings file (one row per booking); out = the 14-night table of extra rooms to sell per hotel; read by = the revenue manager each morning.

hotels.csv
  -> 01_data_cleaning.ipynb   fill agent/company/country/children, clip adr >= 0,
                              add nights, total_guests, arrival_date; drop Bin 2 + Bin 3
  -> hotels_model_ready.csv   (119,390 rows x 28 columns)
  -> split_periods()          src/contract.py: train < 2016-07-01 | choose | proof >= 2017-01-01
  -> GBDT pipeline            Model.ipynb: fit on train, predict_proba per booking
  -> nightly_table()          src/contract.py: sum p, spread, stay value per hotel-night
  -> tune_buffer()            Model.ipynb: pick z on choose, freeze
  -> compute_oversell()       Model.ipynb: max(0, round(expected - z * spread))
  -> night_cost()             src/contract.py: empty rooms, walks, EUR  (src/backtest.py)
  -> 14-night table           Model.ipynb, last cell

### 9. What are the limits?

**Our answer:** Trained and tested on one public-style dataset covering 2015-07 to 2017-08 for two hotels; the proof period is only 8 months (486 hotel-nights). Run time has not been measured yet.
The EUR 300 walk cost and the 10-per-1,000 ceiling are contract assumptions. The ceiling is checked over the whole period, not night by night.
The backtest uses each night's final booking list. In production the list will still be growing 14 days out, so "rooms booked" will be lower than in the test.
No guest names or contact details are in the data; only agent and company ID numbers, stored as text.

### 10. What could go wrong?

**Our answer:**Cancellations cluster. The spread assumes bookings cancel independently. Group or agent bookings that cancel together make real swings larger than the spread, and the chosen buffer is already 0.

Drift. The model has not seen 2017 data in training and uses arrival_date_year as an input; a change in pricing, deposits or channel mix can shift the cancellation rate.

Trust. Walks per 1,000 are 7.78 against 7.42 for the flat rule, slightly higher. One night with many walked guests could make the revenue manager stop using the table.

### 11. What is out of scope?

**Our answer:**Pricing, room-type or guest-level decisions, and any automatic selling: the system only recommends a number.
Live integration with the booking system, and handling of walked guests (rebooking, compensation).
Changing deposit or cancellation policy, and forecasting new demand.
