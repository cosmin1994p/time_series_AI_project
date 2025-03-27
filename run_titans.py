import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from src.data.data_preprocessing import load_and_preprocess_data
from src.features.sequence_builder import create_sequences
from src.models.titans import TitansForecastModel
from src.training.train_model import train_model
from src.evaluation.evaluate_model import evaluate_model
from src.utils.device import get_device

# --- Configs ---
INPUT_WINDOW = 168
OUTPUT_WINDOW = 24
EPOCHS = 5
BATCH_SIZE = 64
DATA_PATH = "data/raw/Consum National 2022-2024.csv"

device = get_device()
df, scaler = load_and_preprocess_data(DATA_PATH)
X, y = create_sequences(df['MWh_scaled'].values, INPUT_WINDOW, OUTPUT_WINDOW)


split_date = df[df['Timestamp'] >= "2024-01-01"].index[0]
split_idx = split_date - INPUT_WINDOW - OUTPUT_WINDOW
X_train, y_train = X[:split_idx], y[:split_idx]
X_test, y_test = X[split_idx:], y[split_idx:]


X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=BATCH_SIZE, shuffle=True)


model = TitansForecastModel()
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
train_model(model, train_loader, loss_fn, optimizer, device, epochs=EPOCHS)


model.eval()
with torch.no_grad():
    preds = model(X_test_tensor.to(device)).cpu().numpy()
    trues = y_test_tensor.cpu().numpy()


preds = scaler.inverse_transform(preds)
trues = scaler.inverse_transform(trues)

metrics = evaluate_model(trues, preds)
print("\n📊 Forecast Metrics:")
for k, v in metrics.items():
    print(f"{k:<6} = {v:.4f}")
