"""
DeepAQI — Delhi Air Quality Forecasting — Streamlit Application

Loads pre-trained GRU / LSTM models and predicts the next-hour AQI value
from uploaded air quality data (supports full multi-station data with
automatic aggregation).
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import tensorflow as tf
from tensorflow import keras

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_DIR = "model"
GRU_PATH = os.path.join(MODEL_DIR, "gru_model.keras")
LSTM_PATH = os.path.join(MODEL_DIR, "lstm_model.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.pkl")

AQI_CATEGORIES = [
    (0, 50, "Good", "#2ecc71"),
    (51, 100, "Satisfactory", "#27ae60"),
    (101, 200, "Moderate", "#f39c12"),
    (201, 300, "Poor", "#e67e22"),
    (301, 400, "Very Poor", "#e74c3c"),
    (401, 500, "Severe", "#8e44ad"),
]


def get_aqi_category(value):
    """Return (label, color) for a given AQI value."""
    for low, high, label, color in AQI_CATEGORIES:
        if low <= value <= high:
            return label, color
    if value > 500:
        return "Severe", "#8e44ad"
    return "Unknown", "#95a5a6"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_artifacts():
    """Load models, scalers, and metadata from disk."""
    errors = []
    for path, name in [(GRU_PATH, "GRU model"), (LSTM_PATH, "LSTM model"),
                       (SCALER_PATH, "Scaler"), (METADATA_PATH, "Metadata")]:
        if not os.path.exists(path):
            errors.append(f"{name} not found at `{path}`")
    if errors:
        return None, errors

    scalers = joblib.load(SCALER_PATH)
    metadata = joblib.load(METADATA_PATH)
    gru_model = keras.models.load_model(GRU_PATH)
    lstm_model = keras.models.load_model(LSTM_PATH)

    return {
        "gru": gru_model,
        "lstm": lstm_model,
        "scaler_X": scalers["scaler_X"],
        "scaler_y": scalers["scaler_y"],
        "metadata": metadata,
    }, []


def preprocess_input(df, metadata, scaler_X):
    """Apply the same feature engineering and scaling used during training.

    Handles multi-station data by aggregating across stations per hour,
    matching the training pipeline.

    Returns the scaled feature array and the processed DataFrame.
    """
    df = df.copy()

    # Ensure datetime
    if "event_timestamp" in df.columns:
        df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
        df = df.sort_values("event_timestamp").reset_index(drop=True)

    # Drop non-numeric identifiers
    for col in ["location_id", "city"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Aggregate across stations per hour (same as training pipeline)
    if "event_timestamp" in df.columns:
        df["event_timestamp"] = df["event_timestamp"].dt.floor("h")
        df = df.groupby("event_timestamp").mean().reset_index()
        df = df.sort_values("event_timestamp").reset_index(drop=True)

    target = metadata["target"]

    # Fill missing values
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[num_cols] = df[num_cols].ffill().bfill()

    # Temporal features
    if "event_timestamp" in df.columns:
        df["hour"] = df["event_timestamp"].dt.hour
        df["day_of_week"] = df["event_timestamp"].dt.dayofweek
        df["month"] = df["event_timestamp"].dt.month

    # Lag features
    if target in df.columns:
        df["aqi_lag_1"] = df[target].shift(1)
        df["aqi_lag_24"] = df[target].shift(24)
        df["aqi_rolling_mean_24"] = df[target].shift(1).rolling(window=24).mean()
        df["aqi_rolling_std_24"] = df[target].shift(1).rolling(window=24).std()

    df = df.dropna().reset_index(drop=True)

    features = metadata["features"]
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required columns: {missing_features}")

    X = df[features].values
    X_scaled = scaler_X.transform(X)
    return X_scaled, df


def create_sequences(X, seq_len):
    """Create sequences for the last prediction window."""
    if len(X) < seq_len:
        raise ValueError(
            f"Need at least {seq_len} rows after preprocessing, got {len(X)}."
        )
    # Use last seq_len rows to predict the next hour
    return X[-seq_len:].reshape(1, seq_len, X.shape[1])


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="DeepAQI — Delhi AQI Forecasting",
        page_icon="🌫️",
        layout="wide",
    )

    st.title("🌫️ DeepAQI — Delhi Air Quality Forecasting")
    st.markdown(
        "Upload recent air quality data and get a **next-hour AQI prediction** "
        "using a pre-trained GRU or LSTM deep learning model trained on data "
        "from **14 monitoring stations** across Delhi."
    )

    # Load artifacts
    artifacts, errors = load_artifacts()
    if errors:
        st.error("❌ Model artifacts are missing. Please train models first using the notebook.")
        for e in errors:
            st.warning(e)
        st.stop()

    metadata = artifacts["metadata"]
    seq_len = metadata["seq_len"]

    # --- Sidebar ---
    st.sidebar.header("⚙️ Settings")
    model_choice = st.sidebar.selectbox("Select Model", ["GRU", "LSTM"])

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Model Details**")
    st.sidebar.markdown(f"- Architecture: 3-layer stacked {model_choice}")
    st.sidebar.markdown(f"- Sequence length: {seq_len} hours")
    st.sidebar.markdown(f"- Features: {len(metadata['features'])}")
    st.sidebar.markdown(f"- Target: `{metadata['target']}`")
    st.sidebar.markdown(f"- Regularization: L2 + Dropout + Gradient Clipping")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Dataset Info**")
    st.sidebar.markdown("- 14 monitoring stations aggregated")
    st.sidebar.markdown("- Hourly city-wide averages")
    st.sidebar.markdown("- Source: [HuggingFace](https://huggingface.co/datasets/abhinavsarkar/delhi_air_quality_feature_store_processed.csv)")

    # --- File upload ---
    st.header("📤 1. Upload Data")
    st.markdown(
        f"Upload a CSV or Parquet file with at least **{seq_len + 25}** recent "
        "hourly observations. The file must contain pollutant readings and weather "
        "data columns. Multi-station data is automatically aggregated."
    )
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "parquet"])

    if uploaded_file is not None:
        # Load data
        try:
            if uploaded_file.name.endswith(".parquet"):
                df = pd.read_parquet(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Failed to load file: {e}")
            st.stop()

        st.success(f"✅ Loaded {len(df):,} rows and {len(df.columns)} columns.")

        # Preview
        with st.expander("Preview raw data"):
            st.dataframe(df.tail(20))

        # Preprocess
        try:
            X_scaled, df_processed = preprocess_input(
                df, metadata, artifacts["scaler_X"]
            )
        except ValueError as e:
            st.error(str(e))
            st.stop()

        st.info(f"After preprocessing: {len(df_processed):,} rows, {X_scaled.shape[1]} features.")

        # --- Historical visualization ---
        st.header("📈 2. Recent AQI History")
        target = metadata["target"]
        if target in df_processed.columns and "event_timestamp" in df_processed.columns:
            chart_data = df_processed[["event_timestamp", target]].tail(500)
            st.line_chart(chart_data.set_index("event_timestamp")[target])
        elif target in df_processed.columns:
            st.line_chart(df_processed[target].tail(500))

        # --- Prediction ---
        st.header("🎯 3. Next-Hour AQI Prediction")

        try:
            X_seq = create_sequences(X_scaled, seq_len)
        except ValueError as e:
            st.error(str(e))
            st.stop()

        model = artifacts["gru"] if model_choice == "GRU" else artifacts["lstm"]
        pred_scaled = model.predict(X_seq, verbose=0).flatten()
        pred_aqi = artifacts["scaler_y"].inverse_transform(
            pred_scaled.reshape(-1, 1)
        ).flatten()[0]

        pred_aqi = max(0, float(pred_aqi))
        category, color = get_aqi_category(pred_aqi)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Predicted AQI", value=f"{pred_aqi:.1f}")
        with col2:
            st.markdown(
                f"<div style='background-color:{color};color:white;"
                f"padding:18px;border-radius:10px;text-align:center;"
                f"font-size:22px;font-weight:bold;'>{category}</div>",
                unsafe_allow_html=True,
            )
        with col3:
            st.metric(label="Model Used", value=model_choice)

        # AQI legend
        st.markdown("---")
        st.subheader("AQI Categories")
        legend_cols = st.columns(len(AQI_CATEGORIES))
        for i, (low, high, label, c) in enumerate(AQI_CATEGORIES):
            with legend_cols[i]:
                st.markdown(
                    f"<div style='background-color:{c};color:white;"
                    f"padding:8px;border-radius:6px;text-align:center;"
                    f"font-size:13px;'><b>{label}</b><br>{low}-{high}</div>",
                    unsafe_allow_html=True,
                )

    else:
        st.info("👆 Please upload a data file to get started.")


if __name__ == "__main__":
    main()
