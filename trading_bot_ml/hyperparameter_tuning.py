import itertools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import joblib

# Load data
df = pd.read_csv('prepared_features.csv')
X = df.drop(columns=['target']).values
y = df['target'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'scaler.pkl')

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Model class
class TradingModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# Grid search
param_grid = {
    'hidden_size': [32, 64, 128],
    'lr': [0.01, 0.001]
}

for hidden_size, lr in itertools.product(param_grid['hidden_size'], param_grid['lr']):
    model = TradingModel(input_size=X_train.shape[1], hidden_size=hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)

    for epoch in range(30):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    # Evaluate
    model.eval()
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)
    val_outputs = model(X_val_tensor)
    val_preds = torch.argmax(val_outputs, dim=1).cpu().numpy()
    acc = accuracy_score(y_val, val_preds)
    print(f"Hidden: {hidden_size}, LR: {lr}, Accuracy: {acc*100:.2f}%")