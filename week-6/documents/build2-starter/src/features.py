
import pandas as pd

# Only approved Bin 1 features
FEATURE_COLUMNS = [
    "hotel",
    "lead_time",
    "arrival_date_year",
    "arrival_date_month",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "country",
    "market_segment",
    "distribution_channel",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "agent",
    "company",
    "days_in_waiting_list",
    "customer_type"
]


def build_features(bookings):

    if not isinstance(bookings, pd.DataFrame):
        raise TypeError("bookings must be a pandas DataFrame")

    df = bookings.copy()

    # Select only approved features
    missing_columns = [
        col for col in FEATURE_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    X = df[FEATURE_COLUMNS].copy()

    return X
