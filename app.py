import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from data_utils.loader import load_time_series
from forecasting.forecaster import train_and_forecast
from anomalies.detector import detect_anomalies
from llm.explainer import explain_time_series, explain_model_behavior


st.set_page_config(
    page_title="DeepTime Studio",
    layout="wide"
)

st.title("DeepTime Studio 🧠📈")
st.markdown(
    "Deep learning–powered time series forecasting and anomaly detection "
    "with LSTM + LLM explanations."
)

uploaded_file = st.file_uploader("Upload a time series file (CSV)", type=["csv"])

with st.sidebar:
    st.header("Configuration")

    date_col = st.text_input("Date column name", value="date")
    value_col = st.text_input("Value column name", value="value")

    test_size = st.slider("Test size (fraction)", 0.1, 0.5, 0.2, 0.05)
    forecast_horizon = st.slider("Forecast horizon (steps)", 5, 60, 20, 5)
    seq_length = st.slider("Sequence length (LSTM input window)", 5, 100, 30, 5)
    num_epochs = st.slider("Training epochs", 5, 100, 20, 5)
    anomaly_z = st.slider("Anomaly Z-score threshold", 1.5, 4.0, 2.5, 0.1)

if uploaded_file is None:
    st.info("Upload a CSV file with at least a date column and a numeric value column.")
    st.stop()

# Load data
try:
    df = load_time_series(uploaded_file, date_col=date_col, value_col=value_col)
except Exception as e:
    st.error(f"Error loading file: {e}")
    st.stop()

# Basic sanity checks
n_rows = len(df)
if n_rows < 30:
    st.error(
        f"Your dataset has only {n_rows} rows. "
        "Please upload a time series with at least 30 rows for meaningful training."
    )
    st.stop()

# Compute train/test sizes
train_len = int(n_rows * (1 - test_size))
test_len = n_rows - train_len

if train_len <= seq_length:
    st.error(
        f"Training set is too small for the selected sequence length.\n\n"
        f"- Total rows: {n_rows}\n"
        f"- Train rows: {train_len}\n"
        f"- Sequence length: {seq_length}\n\n"
        "Please reduce the sequence length or the test size, or upload more data."
    )
    st.stop()

if test_len <= seq_length:
    st.error(
        f"Test set is too small for the selected sequence length.\n\n"
        f"- Total rows: {n_rows}\n"
        f"- Test rows: {test_len}\n"
        f"- Sequence length: {seq_length}\n\n"
        "Please reduce the sequence length or the test size, or upload more data."
    )
    st.stop()

if forecast_horizon <= 0:
    st.error("Forecast horizon must be greater than 0.")
    st.stop()

st.subheader("Raw Time Series")
st.dataframe(df.head())

fig = go.Figure()
fig.add_trace(go.Scatter(x=df[date_col], y=df[value_col], mode="lines", name="Value"))
fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig, use_container_width=True)

# Train + forecast
with st.spinner("Training LSTM model and generating forecast..."):
    results = train_and_forecast(
        df,
        date_col=date_col,
        value_col=value_col,
        test_size=test_size,
        forecast_horizon=forecast_horizon,
        seq_length=seq_length,
        num_epochs=num_epochs,
    )

train_df = results["train_df"]
test_df = results["test_df"]
forecast_df = results["forecast_df"]
metrics = results["metrics"]

st.subheader("Model Performance")
st.write(metrics)

# Plot train, test, forecast
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=train_df[date_col], y=train_df[value_col],
                          mode="lines", name="Train"))
fig2.add_trace(go.Scatter(x=test_df[date_col], y=test_df[value_col],
                          mode="lines", name="Test"))
fig2.add_trace(go.Scatter(x=forecast_df[date_col], y=forecast_df["forecast"],
                          mode="lines", name="Forecast"))
fig2.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10))
st.subheader("Train / Test / Forecast")
st.plotly_chart(fig2, use_container_width=True)

# Anomaly detection on full series
with st.spinner("Detecting anomalies..."):
    anomalies_df = detect_anomalies(df, value_col=value_col, z_threshold=anomaly_z)

st.subheader("Anomalies")
if anomalies_df.empty:
    st.write("No anomalies detected at the current threshold.")
else:
    st.write(f"Detected {len(anomalies_df)} anomalies.")
    st.dataframe(anomalies_df.head())

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df[date_col], y=df[value_col],
                              mode="lines", name="Value"))
    fig3.add_trace(go.Scatter(
        x=anomalies_df[date_col],
        y=anomalies_df[value_col],
        mode="markers",
        name="Anomaly",
        marker=dict(color="red", size=8)
    ))
    fig3.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig3, use_container_width=True)

# LLM explanations
st.subheader("LLM Explanations")

col1, col2 = st.columns(2)

with col1:
    if st.button("Explain time series behavior"):
        with st.spinner("Generating explanation..."):
            explanation = explain_time_series(df, date_col=date_col, value_col=value_col)
        st.markdown(explanation)

with col2:
    if st.button("Explain model & forecast behavior"):
        with st.spinner("Generating explanation..."):
            explanation = explain_model_behavior(metrics, forecast_df, date_col=date_col)
        st.markdown(explanation)
