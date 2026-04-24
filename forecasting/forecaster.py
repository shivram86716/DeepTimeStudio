import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error

from models.lstm_model import LSTMForecaster


class TimeSeriesDataset(Dataset):
    def __init__(self, series: np.ndarray, seq_length: int):
        self.series = series
        self.seq_length = seq_length

    def __len__(self):
        return max(0, len(self.series) - self.seq_length)

    def __getitem__(self, idx):
        x = self.series[idx: idx + self.seq_length]
        y = self.series[idx + self.seq_length]
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor([y], dtype=torch.float32)
        return x, y


def train_and_forecast(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    test_size: float,
    forecast_horizon: int,
    seq_length: int,
    num_epochs: int,
):
    values = df[value_col].values.astype(float)

    split_idx = int(len(values) * (1 - test_size))
    train_values = values[:split_idx]
    test_values = values[split_idx:]

    train_dataset = TimeSeriesDataset(train_values, seq_length)
    test_dataset = TimeSeriesDataset(test_values, seq_length)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMForecaster(input_size=1, hidden_size=64, num_layers=2, dropout=0.1)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x_batch.size(0)

    # Evaluate on test
    model.eval()
    preds_list = []
    true_list = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            preds = model(x_batch)
            preds_list.extend(preds.cpu().numpy().flatten().tolist())
            true_list.extend(y_batch.cpu().numpy().flatten().tolist())

    if len(true_list) == 0:
        mae = float("nan")
        rmse = float("nan")
    else:
        mae = mean_absolute_error(true_list, preds_list)
        rmse = mean_squared_error(true_list, preds_list, squared=False)

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "Test samples": len(true_list),
    }

    # Build train/test DataFrames
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    # Forecast future horizon using last seq_length points
    history = values.copy()
    forecast_values = []

    last_seq = history[-seq_length:].astype(float)

    model.eval()
    with torch.no_grad():
        for _ in range(forecast_horizon):
            x = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
            pred = model(x).cpu().numpy().flatten()[0]
            forecast_values.append(pred)
            last_seq = np.roll(last_seq, -1)
            last_seq[-1] = pred

    last_date = df[date_col].iloc[-1]
    freq = pd.infer_freq(df[date_col])
    if freq is None:
        # Fallback: daily
        freq = "D"

    forecast_index = pd.date_range(start=last_date, periods=forecast_horizon + 1, freq=freq)[1:]
    forecast_df = pd.DataFrame({
        date_col: forecast_index,
        "forecast": forecast_values,
    })

    return {
        "train_df": train_df,
        "test_df": test_df,
        "forecast_df": forecast_df,
        "metrics": metrics,
    }
