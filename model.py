import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib

FEATURES = ['open','high','low','close','volume','rsi','macd','macd_signal',
            'bb_pct','ema20','ema50','atr','vol_ma','returns','stoch','willr','cci','mfi',
            'dist_ema20', 'dist_ema50', 'vp_trend']

class TimeSeriesTransformer(nn.Module):
    def __init__(self, n_features, d_model=64, nhead=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_linear = nn.Linear(n_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_price = nn.Linear(d_model, 1)
        self.fc_dir = nn.Linear(d_model, 1)
        
    def forward(self, x):
        x = self.input_linear(x)
        out = self.transformer_encoder(x)
        last = out[:, -1, :]
        price = self.fc_price(last)
        dir_prob = torch.sigmoid(self.fc_dir(last))
        return price, dir_prob

class GRUPrice(nn.Module):
    def __init__(self, n_features, hidden=64):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, 2, batch_first=True, dropout=0.2)
        self.fc_price = nn.Linear(hidden, 1)
        self.fc_dir = nn.Linear(hidden, 1)  # sigmoid for direction prob
    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        price = self.fc_price(last)
        dir_prob = torch.sigmoid(self.fc_dir(last))
        return price, dir_prob

def make_sequences(df, seq_len=12, tp_pct=0.015, sl_pct=0.01, horizon=24):
    X, y_price, y_dir = [], [], []
    data = df[FEATURES].values
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    
    for i in range(seq_len, len(df) - horizon):
        X.append(data[i-seq_len:i])
        y_price.append(closes[i+1])
        
        entry = closes[i]
        tp = entry * (1 + tp_pct)
        sl = entry * (1 - sl_pct)
        
        label = 0
        for j in range(i+1, i+horizon+1):
            if highs[j] >= tp:
                label = 1
                break
            if lows[j] <= sl:
                label = 0
                break
        
        y_dir.append(label)
        
    return np.array(X), np.array(y_price), np.array(y_dir)

def train_models(df, seq_len=12, epochs=50):
    # Use Triple Barrier sequence generation
    X, y_price, y_dir = make_sequences(df, seq_len, tp_pct=0.015, sl_pct=0.01, horizon=24)
    split = int(len(X) * 0.8)
    X_tr, X_te = X[:split], X[split:]
    yp_tr, yp_te = y_price[:split], y_price[split:]
    yd_tr, yd_te = y_dir[:split], y_dir[split:]

    # Scale X
    scaler = StandardScaler()
    flat_tr = X_tr.reshape(-1, X_tr.shape[-1])
    scaler.fit(flat_tr)
    X_tr_s = scaler.transform(flat_tr).reshape(X_tr.shape)
    X_te_s = scaler.transform(X_te.reshape(-1, X_te.shape[-1])).reshape(X_te.shape)

    # Scale y_price
    price_scaler = StandardScaler()
    yp_tr_s = price_scaler.fit_transform(yp_tr.reshape(-1, 1)).flatten()

    # Transformer for price and direction
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TimeSeriesTransformer(len(FEATURES)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn_p = nn.MSELoss()
    loss_fn_d = nn.BCELoss()
    
    Xt = torch.tensor(X_tr_s, dtype=torch.float32).to(device)
    yt = torch.tensor(yp_tr_s, dtype=torch.float32).unsqueeze(1).to(device)
    yd_t = torch.tensor(yd_tr, dtype=torch.float32).unsqueeze(1).to(device)
    
    print(f"Training Transformer for {epochs} epochs...")
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        price_pred, dir_pred = model(Xt)
        loss_p = loss_fn_p(price_pred, yt)
        loss_d = loss_fn_d(dir_pred, yd_t)
        loss = loss_p + loss_d
        loss.backward()
        opt.step()
        if ep % 10 == 0:
            print(f"Epoch {ep} | Loss: {loss.item():.4f} (Price: {loss_p.item():.4f}, Dir: {loss_d.item():.4f})")

    # XGBoost for direction ensemble
    X_tr_flat = X_tr.reshape(X_tr.shape[0], -1)
    X_te_flat = X_te.reshape(X_te.shape[0], -1)
    xgb = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, eval_metric='logloss')
    xgb.fit(X_tr_flat, yd_tr)
    acc = xgb.score(X_te_flat, yd_te)
    print(f"XGB direction accuracy: {acc*100:.2f}%")

    # Save
    torch.save(model.state_dict(), 'lstm.pt') # Keeping same name for compatibility
    joblib.dump(xgb, 'xgb.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(price_scaler, 'price_scaler.pkl')
    return acc, model, xgb, scaler