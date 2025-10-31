import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from feature_engineering import engineer_features  # we'll modularize feature engineering!

# === CONFIG ===
DATA_PATH = 'historical_data.csv'
MODEL_PATH = 'trading_model.pth'
SCALER_PATH = 'scaler.pkl'

# === 1️⃣ Generate fresh features ===
df = pd.read_csv(DATA_PATH)
features_df = engineer_features(df)
features_df.to_csv('prepared_features.csv', index=False)

# === 2️⃣ Prepare data ===
X = features_df.drop(columns=['target']).values
y = features_df['target'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, SCALER_PATH)

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# === 3️⃣ Model definition ===
class TradingModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

model = TradingModel(input_size=X_train.shape[1]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# === 4️⃣ Training loop ===
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)

EPOCHS = 50
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

# === 5️⃣ Save model ===
torch.save(model.state_dict(), MODEL_PATH)
print("Model retrained and saved.")