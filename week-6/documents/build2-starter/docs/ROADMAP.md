# What happens after this week?

**Owner:** Build Lead (Data Lead in the team of three). **When:** Friday.

The next three weeks, one line each, with the role that owns the work and an hour estimate. Then the production additions from the build plan.

> Hours are our estimates. Numbers come from the profiling notebook and the model notebook (486 proof-period hotel-nights, 2017-01-01 to 2017-08-31).

| Week | What we would build next | Why it matters to the client, with a number | Owner | Hours |
|---|---|---|---|---|
| 7 | As-of-date backtest: score each night using only the bookings on the books when the 14-night table is made | The median cancellation happens 54 days before arrival and 22.1% within 14 days, so a table made two weeks ahead has seen most of the cancelling. Today's EUR 1.63M proof cost assumes the final bookings, and this test shows whether it holds | Model Lead | 12 |
| 8 | Group-aware uncertainty buffer, then refit on all three periods | Rows with a twin cancel at 58.4% against 26.2%, so cancellations move together and the buffer is too narrow. The best model walks 7.78 per 1,000 against a ceiling of 10, and the cancel rate drifts 35.8%, 36.9%, 38.7% | Model Lead, with the Data Lead | 10 |
| 9 | The revenue manager's table: `nightly.py` on a schedule, saved as a file with an override column and a reason | The client renews only if the module pays for itself. GBDT costs EUR 1.63M against EUR 3.42M for the flat rule, a saving of EUR 1.80M (52.5%), about EUR 3,700 per hotel-night | Build Lead | 14 |

## Which production additions come with it?

| Addition | Owner | Hours |
|---|---|---|
| Input file check before every run: columns, dates, row count, and no bin-3 column | Data Lead | 6 |
| Drift monitor: weekly cancel rate by hotel against the 35.8%, 36.9%, 38.7% history, and walks per 1,000 against the ceiling of 10, with an alert when either is crossed | Model Lead | 8 |
| Scheduled retrain that saves the model version and the frozen buffer | Build Lead | 8 |
| Audit trail for every table: input file, model version, buffer, and who overruled which night | Build Lead | 5 |
| Guest data handling: the scoring file carries no names or contact details, and access is limited | Build Lead | 4 |
