import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# === CONFIGURATION ===
DATA_PATH = 'prepared_features.csv'  # replace with your actual path
MODEL_PATH = 'trading_model.pth'
SCALER_PATH = 'scaler.pkl'  # optional, for consistent scaling in inference

# === LOAD DATA ===
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=['target']).values
y = df['target'].values  # assume target already encoded (0,1,2)

# === SCALE FEATURES ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Optionally save scaler for inference consistency
import joblib
joblib.dump(scaler, SCALER_PATH)

# === TRAIN TEST SPLIT ===
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === TENSORS ===
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)

# === MODEL DEFINITION ===
class TradingModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_classes=3):
        super(TradingModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        return out

model = TradingModel(input_size=X_train.shape[1]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# === TRAINING LOOP ===
EPOCHS = 50
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        model.eval()
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)
        acc = (val_outputs.argmax(1) == y_val_tensor).float().mean().item()
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}, Val Acc: {acc*100:.2f}%")

# === SAVE MODEL ===
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")