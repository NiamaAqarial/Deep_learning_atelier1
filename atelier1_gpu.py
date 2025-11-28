#======================= Atelier 1 Niama Aqarial (GPU Version) ========================#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error, r2_score,
                             accuracy_score, classification_report, f1_score)
import warnings
warnings.filterwarnings('ignore')

# GPU Setup
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda if torch.cuda.is_available() else "N/A")
print("Number of GPUs:", torch.cuda.device_count() if torch.cuda.is_available() else 0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")

# =============================================================================
#                               PARTIE 1 – RÉGRESSION
# =============================================================================
print("PARTIE 1 : Régression – Prédiction du prix de clôture (close)\n")

df_prices = pd.read_csv("prices.csv")
try:
    df_prices = pd.read_csv("prices-split-adjusted.csv")
    print("Fichier détecté : prices-split-adjusted.csv → format date avec heure")
    df_prices['date'] = pd.to_datetime(df_prices['date'])
except:
    df_prices = pd.read_csv("prices.csv")
    print("Fichier détecté : prices.csv → format date simple")
    df_prices['date'] = pd.to_datetime(df_prices['date'], format='%Y-%m-%d')

print(f"Dataset chargé : {df_prices.shape[0]:,} lignes, {df_prices.shape[1]} colonnes\n")
print("Shape :", df_prices.shape)
print(df_prices.head(), "\n")

# 1. EDA
plt.figure(figsize=(16,10))

plt.subplot(2,2,1)
aapl = df_prices[df_prices['symbol']=='AAPL']
plt.plot(aapl['date'], aapl['close'], label='AAPL Close Price')
plt.title('Évolution du prix Apple')
plt.xlabel('Date'); plt.ylabel('Prix ($)')
plt.legend()

plt.subplot(2,2,2)
sns.histplot(df_prices['close'], bins=100, kde=True, color='skyblue')
plt.title('Distribution des prix de clôture')

plt.subplot(2,2,3)
corr = df_prices[['open','high','low','close','volume']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.3f')
plt.title('Corrélation des variables prix')

plt.subplot(2,2,4)
df_prices['year'] = df_prices['date'].dt.year
df_prices.groupby('year')['volume'].mean().plot(kind='bar', color='orange')
plt.title('Volume moyen par année')
plt.tight_layout()
plt.show()

# 2. Préparation des données régression
features = ['open', 'high', 'low', 'volume']
X = df_prices[features].values
y = df_prices['close'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Move tensors to GPU
X_train_t = torch.FloatTensor(X_train).to(device)
X_test_t  = torch.FloatTensor(X_test).to(device)
y_train_t = torch.FloatTensor(y_train).reshape(-1,1).to(device)
y_test_t  = torch.FloatTensor(y_test).reshape(-1,1).to(device)

# DataLoader - data will be moved to GPU in training loop
train_dataset_reg = list(zip(torch.FloatTensor(X_train), torch.FloatTensor(y_train).reshape(-1,1)))
train_loader_reg = DataLoader(train_dataset_reg, batch_size=256, shuffle=True)

# 3. Modèle PyTorch
class Regressor(nn.Module):
    def __init__(self, layers=[128,64,32], dropout=0.3):
        super().__init__()
        net = []
        prev = 4
        for h in layers:
            net += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        net.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*net)
    def forward(self, x): return self.net(x)

# Fonction d'entraînement (GPU)
def train_reg(model, epochs=80, lr=0.001, wd=0):
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.MSELoss()
    train_l, test_l = [], []
    for epoch in range(epochs):
        model.train()
        loss_sum = 0
        for xb,yb in train_loader_reg:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.item()
        model.eval()
        with torch.no_grad():
            test_loss = crit(model(X_test_t), y_test_t).item()
        train_l.append(loss_sum/len(train_loader_reg))
        test_l.append(test_loss)
    return train_l, test_l

# 4. GridSearch manuel
print("\nRecherche des meilleurs hyperparamètres (GridSearch)")
grid = ParameterGrid({
    'layers':   [[128,64,32], [256,128,64], [64,32]],
    'lr':       [0.001, 0.0005, 0.01],
    'dropout':  [0.2, 0.3, 0.4]
})

best_mse = np.inf
best_params = None
best_model_reg = None

for params in grid:
    model = Regressor(layers=params['layers'], dropout=params['dropout'])
    _, test_losses = train_reg(model, epochs=50, lr=params['lr'])
    final_mse = test_losses[-1]
    print(f"Testé → {params}  →  Test MSE = {final_mse:.2f}")
    if final_mse < best_mse:
        best_mse = final_mse
        best_params = params
        best_model_reg = model

print(f"\nMeilleur modèle → {best_params} | Test MSE = {best_mse:.2f}")

# Entraînement final du meilleur modèle
train_loss_final, test_loss_final = train_reg(best_model_reg, epochs=150, lr=best_params['lr'], wd=1e-5)

plt.figure(figsize=(10,4))
plt.plot(train_loss_final, label='Train Loss')
plt.plot(test_loss_final, label='Test Loss')
plt.title('Loss / Epochs – Régression (meilleur modèle)')
plt.xlabel('Epoch'); plt.ylabel('MSE Loss'); plt.legend(); plt.grid()
plt.show()

with torch.no_grad():
    pred_reg = best_model_reg(X_test_t).cpu().numpy().flatten()
y_test_cpu = y_test_t.cpu().numpy().flatten()
print(f"RMSE = {mean_squared_error(y_test_cpu, pred_reg, squared=False):.3f}")
print(f"R²   = {r2_score(y_test_cpu, pred_reg):.4f}")

# 5. Régularisation comparée
class RegReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self,x): return self.net(x)

reg_model = RegReg()
_, test_reg = train_reg(reg_model, epochs=150, lr=0.0005, wd=1e-4)

plt.figure(figsize=(8,5))
plt.plot(test_loss_final, label='Modèle de base')
plt.plot(test_reg, label='Avec BatchNorm + fort Dropout + L2')
plt.title('Comparaison régularisation – Régression')
plt.legend(); plt.grid(); plt.show()
