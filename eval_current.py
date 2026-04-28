import numpy as np
from data_fetcher import fetch_btc_1h
from indicators import add_indicators
from model import make_sequences, FEATURES, TimeSeriesTransformer
import torch, joblib

def run_eval():
    print("Fetching data...")
    df = fetch_btc_1h(limit=1000)
    print("Adding indicators...")
    df = add_indicators(df)
    
    print("Loading models...")
    scaler = joblib.load('scaler.pkl')
    xgb = joblib.load('xgb.pkl')
    model = TimeSeriesTransformer(len(FEATURES))
    model.load_state_dict(torch.load('lstm.pt', map_location='cpu'))
    model.eval()

    X, y_price, y_dir = make_sequences(df, 12)
    X_s = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    X_flat = X.reshape(X.shape[0], -1)

    print("Running predictions...")
    xgb_probs = xgb.predict_proba(X_flat)[:, 1]
    with torch.no_grad():
        _, model_probs = model(torch.tensor(X_s, dtype=torch.float32))
        model_probs = model_probs.squeeze().numpy()
        
    ensemble_probs = (xgb_probs + model_probs) / 2
    preds = (ensemble_probs > 0.5).astype(int)

    correct = (preds == y_dir).sum()
    total = len(y_dir)
    print(f"\n=== EVALUATION ON LATEST {total} CANDLES ===")
    print(f"XGB Accuracy: {(xgb.predict(X_flat) == y_dir).mean()*100:.2f}%")
    model_preds = (model_probs > 0.5).astype(int)
    print(f"Transformer Accuracy: {(model_preds == y_dir).mean()*100:.2f}%")
    print(f"Ensemble Direction accuracy: {correct}/{total} = {correct/total*100:.2f}%")

if __name__ == "__main__":
    run_eval()
