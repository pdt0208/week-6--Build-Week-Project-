

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import project modules
from data import read_raw, arrival_date
from features import build_features


def load_selected_method():
    """
    Load the selected model and buffer from backtesting.
    """
    model_path = PROJECT_ROOT / "models" / "selected_method.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Selected model not found: {model_path}. "
            "Run backtest.py first."
        )

    artifact = joblib.load(model_path)

    required_keys = ["pipeline", "model_name", "buffer"]
    missing = [key for key in required_keys if key not in artifact]

    if missing:
        raise ValueError(
            f"Model artifact is missing keys: {missing}"
        )

    return artifact


def predict_cancellation_probability(bookings, pipeline):
    """
    Predict cancellation probability for every booking.
    """
    X = build_features(bookings)

    probabilities = pipeline.predict_proba(X)

    classes = pipeline.classes_
    class_one_positions = np.where(classes == 1)[0]

    if len(class_one_positions) == 0:
        raise ValueError(
            "The selected model does not contain cancellation class 1."
        )

    cancel_position = class_one_positions[0]

    return probabilities[:, cancel_position]


def calculate_recommendations(bookings, probabilities, buffer):
    """
    Calculate overselling recommendation for each hotel and arrival night.

    Expected cancellations = sum of cancellation probabilities.

    Cancellation spread = sqrt(sum(p * (1-p))).

    Recommended oversell = floor(
        max(0, expected cancellations - buffer * spread)
    )
    """

    df = bookings.copy()

    df["cancel_probability"] = probabilities

    df["variance"] = (
        df["cancel_probability"] *
        (1 - df["cancel_probability"])
    )

    # Group bookings by hotel and arrival date
    recommendations = (
        df.groupby(["hotel", "arrival_date"])
        .agg(
            bookings=("cancel_probability", "size"),
            expected_cancellations=("cancel_probability", "sum"),
            variance=("variance", "sum")
        )
        .reset_index()
    )

    recommendations["cancellation_spread"] = np.sqrt(
        recommendations["variance"]
    )

    # If no buffer is available, recommend zero overselling
    if buffer is None:
        recommendations["recommended_oversell"] = 0
    else:
        recommendations["recommended_oversell"] = np.floor(
            np.maximum(
                0,
                recommendations["expected_cancellations"]
                - float(buffer) *
                recommendations["cancellation_spread"]
            )
        ).astype(int)

    recommendations["recommended_oversell"] = (
        recommendations["recommended_oversell"].astype(int)
    )

    recommendations["total_sell_limit"] = (
        recommendations["bookings"]
        + recommendations["recommended_oversell"]
    )

    recommendations["buffer"] = buffer

    # Remove internal calculation column
    recommendations.drop(columns=["variance"], inplace=True)

    return recommendations.sort_values(
        ["arrival_date", "hotel"]
    ).reset_index(drop=True)


def run_nightly(start_date, number_of_nights):
    """
    Main nightly prediction workflow.
    """

    if number_of_nights <= 0:
        raise ValueError("--nights must be greater than zero.")

    start = pd.to_datetime(start_date)
    end = start + pd.Timedelta(days=number_of_nights)

    print("\nSTEP 1: Loading selected model...")

    artifact = load_selected_method()

    pipeline = artifact["pipeline"]
    model_name = artifact["model_name"]
    buffer = artifact["buffer"]

    print("Selected model:", model_name)
    print("Selected buffer:", buffer)

    print("\nSTEP 2: Loading hotel bookings...")

    bookings = read_raw()

    # Generate arrival dates
    bookings = bookings.copy()
    bookings["arrival_date"] = pd.to_datetime(
        arrival_date(bookings),
        errors="coerce"
    )

    # Remove records without a valid arrival date
    bookings = bookings.dropna(subset=["arrival_date"])

    print("Total records loaded:", len(bookings))

    print("\nSTEP 3: Filtering requested arrival nights...")

    # Select the requested date range
    bookings = bookings[
        (bookings["arrival_date"] >= start) &
        (bookings["arrival_date"] < end)
    ].copy()

    if bookings.empty:
        raise ValueError(
            "No bookings found for the requested date range."
        )

    print("Bookings in requested period:", len(bookings))
    print("Start date:", start.date())
    print("End date (exclusive):", end.date())

    print("\nSTEP 4: Predicting cancellation probabilities...")

    probabilities = predict_cancellation_probability(
        bookings,
        pipeline
    )

    print("Predictions completed.")
    print(
        "Average cancellation probability:",
        round(float(np.mean(probabilities)), 4)
    )

    print("\nSTEP 5: Calculating overselling recommendations...")

    recommendations = calculate_recommendations(
        bookings,
        probabilities,
        buffer
    )

    print("\nSTEP 6: Saving recommendations...")

    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "oversell_recommendations.csv"

    recommendations.to_csv(output_path, index=False)

    print("\nFINAL OVERSELL RECOMMENDATIONS")
    print("--------------------------------")

    print(
        recommendations.to_string(index=False)
    )

    print("\nRecommendations saved successfully:")
    print(output_path)

    print("\nNightly prediction completed successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Generate hotel nightly overselling recommendations."
    )

    parser.add_argument(
        "--from",
        dest="start_date",
        required=True,
        help="Starting arrival date in YYYY-MM-DD format."
    )

    parser.add_argument(
        "--nights",
        type=int,
        required=True,
        help="Number of arrival nights to evaluate."
    )

    args = parser.parse_args()

    run_nightly(
        start_date=args.start_date,
        number_of_nights=args.nights
    )


if __name__ == "__main__":
    main()
