
"""
Hotel Overbooking Project - Model Backtesting

Responsibilities:
1. Load historical hotel booking data.
2. Build model features.
3. Split data into Train, Choose and Proof periods.
4. Train multiple ML models.
5. Compare models and uncertainty buffers using the client contract.
6. Evaluate the selected policy once on the Proof period.
7. Save the selected fitted pipeline and policy.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score

from data import read_raw, arrival_date
from features import build_features, FEATURE_COLUMNS

from contract import (
    split_periods,
    nightly_table,
    night_cost,
    WALK_CEILING
)


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "selected_method.joblib"


# --------------------------------------------------
# 2. FEATURE TYPES
# --------------------------------------------------

NUMERIC_FEATURES = [
    "lead_time",
    "arrival_date_year",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "days_in_waiting_list"
]

CATEGORICAL_FEATURES = [
    "hotel",
    "arrival_date_month",
    "country",
    "market_segment",
    "distribution_channel",
    "agent",
    "company",
    "customer_type"
]


# --------------------------------------------------
# 3. PREPROCESSING PIPELINE
# --------------------------------------------------

def create_preprocessor():

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
    ])

    return preprocessor


# --------------------------------------------------
# 4. CREATE ML MODELS
# --------------------------------------------------

def create_models():

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=50,
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=20,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42
        )

    }

    return models


# --------------------------------------------------
# 5. CREATE MODEL PIPELINE
# --------------------------------------------------

def create_model_pipeline(model):

    pipeline = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("model", model)
    ])

    return pipeline


# --------------------------------------------------
# 6. GET CANCELLATION PROBABILITY
# --------------------------------------------------

def get_cancel_probability(model, X):

    probabilities = model.predict_proba(X)

    classes = list(model.named_steps["model"].classes_)

    if 1 not in classes:
        raise ValueError(
            "Model was not trained with cancellation class 1."
        )

    cancel_index = classes.index(1)

    return probabilities[:, cancel_index]


# --------------------------------------------------
# 7. CREATE NIGHTLY OVERSSELL POLICY
# --------------------------------------------------

def create_oversell(nightly, buffer):

    """
    Calculate oversell rooms using expected cancellations
    and uncertainty spread.

    A higher buffer means a more conservative policy.
    """

    if buffer is None:
        return pd.Series(
            0,
            index=nightly.index,
            dtype=int
        )

    oversell = np.floor(
        nightly["expected"] -
        buffer * nightly["spread"]
    )

    oversell = np.maximum(oversell, 0)

    return pd.Series(
        oversell.astype(int),
        index=nightly.index
    )


# --------------------------------------------------
# 8. LOAD AND PREPARE DATA
# --------------------------------------------------

def prepare_data():

    print("\nLoading hotel booking dataset...")

    df = read_raw()

    print("Original dataset shape:", df.shape)

    # Create actual arrival date from source date columns
    df["arrival_date"] = arrival_date(df)

    # Calculate total stay nights
    df["nights"] = (
        df["stays_in_weekend_nights"] +
        df["stays_in_week_nights"]
    )

    # Ensure required columns have valid values
    df["adr"] = pd.to_numeric(df["adr"], errors="coerce")
    df["is_canceled"] = pd.to_numeric(
        df["is_canceled"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "arrival_date",
            "adr",
            "is_canceled",
            "nights"
        ]
    ).copy()

    # Keep valid binary target values
    df = df[df["is_canceled"].isin([0, 1])].copy()

    df["is_canceled"] = df["is_canceled"].astype(int)

    print("Prepared dataset shape:", df.shape)

    return df


# --------------------------------------------------
# 9. TIME-BASED TRAIN, CHOOSE AND PROOF SPLIT
# --------------------------------------------------

def create_splits(df):

    train, choose, proof = split_periods(df)

    print("\nTime-based data split:")
    print("Train rows:", len(train))
    print("Choose rows:", len(choose))
    print("Proof rows:", len(proof))

    if train.empty or choose.empty or proof.empty:
        raise ValueError(
            "One of the Train, Choose or Proof periods is empty."
        )

    return train, choose, proof


# --------------------------------------------------
# 10. EVALUATE ONE MODEL AND BUFFER ON CHOOSE
# --------------------------------------------------

def evaluate_policy(bookings, probabilities, buffer):

    nightly = nightly_table(bookings, probabilities)

    oversell = create_oversell(nightly, buffer)

    result = night_cost(nightly, oversell)

    return result


# --------------------------------------------------
# 11. TRAIN AND COMPARE MODELS
# --------------------------------------------------

def train_and_compare(train, choose):

    print("\nStarting model training...")

    X_train = build_features(train)
    y_train = train["is_canceled"]

    X_choose = build_features(choose)
    y_choose = choose["is_canceled"]

    model_results = []
    fitted_models = {}

    models = create_models()

    # Candidate uncertainty buffers in standard deviations.
    # Zero oversell is also tested as a safe baseline policy.
    buffers = list(np.arange(-1.0, 10.01, 0.25))
    buffers.append(None)

    for model_name, estimator in models.items():

        print(f"\nTraining: {model_name}")

        pipeline = create_model_pipeline(estimator)

        # Fit preprocessing and model on Train only
        pipeline.fit(X_train, y_train)

        fitted_models[model_name] = pipeline

        # Predict cancellation probabilities on Choose
        probabilities = get_cancel_probability(
            pipeline,
            X_choose
        )

        if y_choose.nunique() == 2:
            auc = roc_auc_score(
                y_choose,
                probabilities
            )
        else:
            auc = np.nan

        print("Choose AUC:", round(auc, 4) if not np.isnan(auc) else "N/A")

        # Evaluate each policy on Choose
        for buffer in buffers:

            result = evaluate_policy(
                choose,
                probabilities,
                buffer
            )

            model_results.append({

                "model": model_name,
                "buffer": buffer,
                "auc": auc,
                "cost_eur": result["cost_eur"],
                "empty_rooms": result["empty_rooms"],
                "walked_guests": result["walked_guests"],
                "walks_per_1000": result["walks_per_1000"],
                "ceiling_met": result["ceiling_met"]

            })

    results_df = pd.DataFrame(model_results)

    results_path = OUTPUT_DIR / "model_comparison.csv"

    results_df.to_csv(results_path, index=False)

    print("\nModel comparison saved:", results_path)

    return results_df, fitted_models


# --------------------------------------------------
# 12. SELECT BEST ACCEPTABLE MODEL AND BUFFER
# --------------------------------------------------

def select_best_model(results_df, fitted_models):

    acceptable = results_df[
        results_df["ceiling_met"] == True
    ].copy()

    if acceptable.empty:

        raise RuntimeError(
            "No candidate policy met the client's walking ceiling. "
            "Do not select an unacceptable policy."
        )

    # Lowest financial cost among acceptable policies
    acceptable = acceptable.sort_values(
        by=["cost_eur", "walks_per_1000"],
        ascending=[True, True]
    )

    best = acceptable.iloc[0]

    best_model_name = best["model"]
    best_buffer = best["buffer"]

    best_model = fitted_models[best_model_name]

    print("\nSelected model:", best_model_name)
    print("Selected buffer:", best_buffer)
    print("Choose cost:", best["cost_eur"])
    print("Choose walks per 1000:", best["walks_per_1000"])

    return best_model, best_model_name, best_buffer


# --------------------------------------------------
# 13. FINAL PROOF EVALUATION - RUN ONCE
# --------------------------------------------------

def evaluate_proof(best_model, best_buffer, proof):

    print("\nRunning final Proof evaluation...")

    X_proof = build_features(proof)

    probabilities = get_cancel_probability(
        best_model,
        X_proof
    )

    # The Proof period is evaluated only once,
    # using the model and buffer selected on Choose.
    proof_result = evaluate_policy(
        proof,
        probabilities,
        best_buffer
    )

    proof_df = pd.DataFrame([proof_result])

    proof_path = OUTPUT_DIR / "proof_results.csv"

    proof_df.to_csv(proof_path, index=False)

    print("\nFINAL PROOF RESULTS")
    print("-------------------")

    for key, value in proof_result.items():
        print(f"{key}: {value}")

    print("\nProof results saved:", proof_path)

    return proof_result


# --------------------------------------------------
# 14. SAVE SELECTED MODEL AND POLICY
# --------------------------------------------------

def save_selected_model(
    best_model,
    best_model_name,
    best_buffer
):

    artifact = {

        "pipeline": best_model,
        "model_name": best_model_name,
        "buffer": best_buffer,
        "feature_columns": FEATURE_COLUMNS

    }

    joblib.dump(artifact, MODEL_PATH)

    print("\nSelected model saved:", MODEL_PATH)


# --------------------------------------------------
# 15. MAIN FUNCTION
# --------------------------------------------------

def main():

    print("=" * 55)
    print("HOTEL OVERBOOKING - BACKTEST")
    print("=" * 55)

    # Load and prepare dataset
    df = prepare_data()

    # Create time-based splits
    train, choose, proof = create_splits(df)

    # Train and compare models using Train and Choose
    results_df, fitted_models = train_and_compare(
        train,
        choose
    )

    # Select model and policy using Choose only
    best_model, best_model_name, best_buffer = select_best_model(
        results_df,
        fitted_models
    )

    # Evaluate the selected model and policy on Proof once
    proof_result = evaluate_proof(
        best_model,
        best_buffer,
        proof
    )

    # Save fitted pipeline and selected policy
    save_selected_model(
        best_model,
        best_model_name,
        best_buffer
    )

    print("\nBacktesting completed successfully.")


if __name__ == "__main__":
    main()
