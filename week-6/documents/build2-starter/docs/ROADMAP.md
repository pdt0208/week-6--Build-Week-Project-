# What happens after this week?

**Owner:** Build Lead  
**When:** Friday

The next three weeks focus on improving the reliability of nightly recommendations, connecting the system to hotel booking data, and making the results easier for hotel teams to use.

| Week | What we would build next | Why it matters to the client, with a number | Owner | Hours |
|---|---|---|---|---:|
| 7 | Create booking-date snapshots so the system can use only the bookings available at a particular point in time. | Reduces the limitation of using final booking data and makes the 14-day recommendation closer to a genuine advance forecast. | Data Lead | 10 |
| 8 | Add model monitoring and evaluate prediction performance and cancellation-probability calibration over time. | Helps detect when model performance changes and protects the client's maximum limit of 10 walked guests per 1,000 arrivals. | Model Lead | 8 |
| 9 | Build a dashboard to display nightly recommendations, expected cancellations, buffer, and oversell decisions. | Makes recommendations easier for hotel staff to understand and review for each arrival night. | Build Lead | 10 |

## Which production additions come with it?

| Addition | Owner | Hours |
|---|---|---:|
| Live Property Management System (PMS) integration | Build Lead | 12 |
| Booking-date snapshots | Data Lead | 10 |
| Model performance monitoring | Model Lead | 8 |
| Automated model retraining | Model Lead | 8 |
| Recommendation dashboard | Build Lead | 10 |
| Manual override and decision logging | Build Lead | 6 |

These are estimated development hours, not completed work. The production additions would be prioritised after the current MVP is tested and accepted. The walk ceiling of 10 guests per 1,000 arrivals remains a mandatory constraint for every future version.


