import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from sklearn.preprocessing import MinMaxScaler

# ── Model definition (must match training) ──────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, batch_first=True)
        self.fc   = nn.Linear(50, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ── App startup: train a fresh model on the sine wave ───────────────────────
app = FastAPI(
    title="LSTM Time Series Forecast API",
    description="Accepts 10 numeric values and returns the next predicted value.",
    version="1.0.0",
)

SEQ_LEN = 10
scaler  = MinMaxScaler()
model   = LSTMModel()

def _train():
    data = np.sin(np.linspace(0, 50, 200)).reshape(-1, 1)
    data_scaled = scaler.fit_transform(data)

    xs, ys = [], []
    for i in range(len(data_scaled) - SEQ_LEN):
        xs.append(data_scaled[i:i+SEQ_LEN])
        ys.append(data_scaled[i+SEQ_LEN])
    X = torch.tensor(np.array(xs), dtype=torch.float32)
    y = torch.tensor(np.array(ys), dtype=torch.float32)

    loss_fn   = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for _ in range(100):
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()

_train()  # runs once when the service starts

# ── Request / Response schemas ───────────────────────────────────────────────
class ForecastRequest(BaseModel):
    values: List[float] = Field(
        ...,
        min_length=SEQ_LEN,
        max_length=SEQ_LEN,
        example=[0.0, 0.24, 0.47, 0.68, 0.84, 0.95, 0.99, 0.96, 0.86, 0.71],
        description=f"Exactly {SEQ_LEN} recent time-series values.",
    )

class ForecastResponse(BaseModel):
    input_values: List[float]
    predicted_next_value: float

# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/", summary="Health check")
def root():
    return {"status": "ok", "message": "LSTM Forecast API is running."}

@app.post("/predict", response_model=ForecastResponse, summary="Predict next value")
def predict(req: ForecastRequest):
    if len(req.values) != SEQ_LEN:
        raise HTTPException(400, f"Provide exactly {SEQ_LEN} values.")

    arr     = np.array(req.values, dtype=np.float32).reshape(-1, 1)
    scaled  = scaler.transform(arr).reshape(1, SEQ_LEN, 1)
    tensor  = torch.tensor(scaled, dtype=torch.float32)

    with torch.no_grad():
        pred_scaled = model(tensor).item()

    pred_original = scaler.inverse_transform([[pred_scaled]])[0][0]

    return ForecastResponse(
        input_values=req.values,
        predicted_next_value=round(float(pred_original), 6),
    )
