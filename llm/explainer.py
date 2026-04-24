import os
import pandas as pd
from groq import Groq


def _get_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable is not set.")
    return Groq(api_key=api_key)


def explain_time_series(df: pd.DataFrame, date_col: str, value_col: str) -> str:
    client = _get_client()

    sample = df.tail(50).to_dict(orient="records")

    prompt = f"""
You are a senior data scientist. You are given a univariate time series.

The last 50 points are:
{sample}

Column '{date_col}' is the timestamp, and '{value_col}' is the numeric value.

Explain the overall behavior of this time series in clear, non-technical language:
- Trends
- Seasonality (if any)
- Volatility
- Any noticeable shifts or patterns

Keep it under 250 words.
"""

    chat_completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are an expert time series analyst."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
    )

    return chat_completion.choices[0].message.content


def explain_model_behavior(metrics: dict, forecast_df: pd.DataFrame, date_col: str) -> str:
    client = _get_client()

    sample_forecast = forecast_df.head(20).to_dict(orient="records")

    prompt = f"""
You are a senior ML engineer. You are given:

Model metrics:
{metrics}

And the first 20 forecasted points:
{sample_forecast}

Column '{date_col}' is the timestamp, and 'forecast' is the predicted value.

Explain:
- How well the model is performing
- How to interpret the forecast
- Any caveats or limitations
- How a business stakeholder should use these results

Keep it under 250 words, clear and practical.
"""

    chat_completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are an expert ML engineer and communicator."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
    )

    return chat_completion.choices[0].message.content
