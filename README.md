# DeepTime Studio 🧠📈

DeepTime Studio is a deep learning–powered time series forecasting and anomaly detection app built with Streamlit, PyTorch, and Groq LLMs.

Upload a time series (CSV) and the app will:

- Clean and parse your time series
- Train an LSTM model for forecasting
- Evaluate performance (MAE, RMSE)
- Generate future forecasts
- Detect anomalies using Z-score
- Use an LLM to explain time series behavior and model performance in natural language

---

## 🧱 Architecture

```text
DeepTime Studio
│
├── app.py                     # Main Streamlit app + validation/safety checks
│
├── data_utils/
│   └── loader.py              # Time series loading & preprocessing
│
├── models/
│   └── lstm_model.py          # PyTorch LSTM forecaster
│
├── forecasting/
│   └── forecaster.py          # Training loop + forecasting logic
│
├── anomalies/
│   └── detector.py            # Z-score based anomaly detection
│
└── llm/
    └── explainer.py           # Groq LLM-based explanations
```

## 🧱 Architecture Diagram
```text
+----------------------+
|      User (UI)       |
+----------+-----------+
           |
           v
+----------------------+        +----------------------+
|       app.py         |------->|  data_utils.loader   |
|  Streamlit frontend  |        |  load_time_series    |
+----------+-----------+        +----------------------+
           |
           | calls
           v
+----------------------+        +----------------------+
| forecasting.forecaster|<---->|   models.lstm_model  |
|  train_and_forecast   |      |   LSTMForecaster     |
+----------+-----------+        +----------------------+
           |
           | uses
           v
+----------------------+        +----------------------+
| anomalies.detector   |        |    llm.explainer     |
| detect_anomalies     |        | explain_* functions  |
+----------+-----------+        +----------------------+
           |
           v
+----------------------+
|  Streamlit outputs   |
|  charts, tables, LLM |
|  explanations        |
+----------------------+
```


## 🧱 System flow Diagram

text```
[1] User uploads CSV
        |
        v
[2] app.py
    - Reads config from sidebar
    - Calls load_time_series()
        |
        v
[3] loader.py
    - Validates columns
    - Parses dates
    - Sorts by date
    - Ensures numeric values
        |
        v
[4] app.py safety checks
    - Check row count
    - Check train/test sizes vs seq_length
    - Check forecast horizon
    - If invalid -> st.error + st.stop
        |
        v
[5] train_and_forecast()
    - Build TimeSeriesDataset
    - Train LSTMForecaster
    - Evaluate (MAE, RMSE)
    - Generate future forecast
        |
        v
[6] detect_anomalies()
    - Compute Z-scores
    - Flag anomalies
        |
        v
[7] app.py visualizations
    - Raw series
    - Train/Test/Forecast
    - Anomalies
        |
        v
[8] LLM explanations
    - explain_time_series()
    - explain_model_behavior()
        |
        v
[9] Streamlit UI
    - User reads charts + explanations
```