import numpy as np
import pandas as pd
import torch, joblib
from data_fetcher import fetch_btc_1h
from indicators import add_indicators, find_sr_zones, detect_market_structure
from confluence import calculate_confluence_score
from model import TimeSeriesTransformer, FEATURES

def run_confluence_backtest():
    print("=== MULTI-LAYER CONFLUENCE BACKTEST ===")
    print("Fetching 5000 candles for robust test...")
    df = fetch_btc_1h(limit=5000)
    
    print("Applying indicators...")
    df = add_indicators(df)
    supports, resistances = find_sr_zones(df, window=20)
    
    # We will simulate 15m/4h using shifted 1H data to save complexity in backtest
    # This is a proxy backtest. A true tick-level backtest requires granular DB data.
    
    try:
        scaler = joblib.load('scaler.pkl')
        xgb = joblib.load('xgb.pkl')
        model = TimeSeriesTransformer(len(FEATURES))
        model.load_state_dict(torch.load('lstm.pt', map_location='cpu'))
        model.eval()
    except Exception as e:
        print(f"Error loading models: {e}. Please run train_models() first if needed.")
        return

    wins = 0
    losses = 0
    skipped = 0
    
    # Walk forward
    print("Running walk-forward evaluation...")
    for i in range(100, len(df) - 24):
        # Slice data up to i
        current_df = df.iloc[:i]
        current_price = current_df['close'].iloc[-1]
        
        structure = detect_market_structure(current_df, window=10)
        
        # Proxy 15m and 4h using current 1h (for backtest simplicity)
        score, conf_dir, _ = calculate_confluence_score(
            current_df, current_df, current_df, current_price, supports, resistances, structure
        )
        
        if score < 4 or conf_dir == "NEUTRAL":
            skipped += 1
            continue
            
        # ML check
        seq = current_df[FEATURES].values[-12:]
        if len(seq) < 12: continue
        
        seq_s = scaler.transform(seq).reshape(1, 12, -1)
        xgb_prob = xgb.predict_proba(seq.reshape(1, -1))[0][1]
        with torch.no_grad():
            _, trans_prob = model(torch.tensor(seq_s, dtype=torch.float32))
            trans_prob = trans_prob.item()
            
        ml_prob = (xgb_prob + trans_prob) / 2
        ml_dir = "LONG" if ml_prob > 0.5 else "SHORT"
        
        if conf_dir != ml_dir:
            skipped += 1
            continue
            
        # Trade logic (simplified 1.5% SL, 3% TP)
        sl = current_price * 0.985 if conf_dir == "LONG" else current_price * 1.015
        tp = current_price * 1.03 if conf_dir == "LONG" else current_price * 0.97
        
        # Look ahead 24 hours to see outcome
        future_df = df.iloc[i+1:i+25]
        trade_result = None
        
        for _, row in future_df.iterrows():
            if conf_dir == "LONG":
                if row['low'] <= sl: trade_result = "LOSS"; break
                if row['high'] >= tp: trade_result = "WIN"; break
            else:
                if row['high'] >= sl: trade_result = "LOSS"; break
                if row['low'] <= tp: trade_result = "WIN"; break
                
        if trade_result == "WIN": wins += 1
        elif trade_result == "LOSS": losses += 1
        
    total_trades = wins + losses
    if total_trades > 0:
        win_rate = wins / total_trades * 100
        print(f"\nRESULTS:")
        print(f"Total Evaluated: {len(df)-124}")
        print(f"Trades Taken: {total_trades} (Skipped: {skipped})")
        print(f"Wins: {wins} | Losses: {losses}")
        print(f"Win Rate: {win_rate:.2f}%")
        
        if win_rate >= 75:
            print("🟢 TARGET ACHIEVED (>75% Accuracy)")
        elif win_rate >= 65:
            print("🟡 ACCEPTABLE (>65% Accuracy) - Try tweaking confluence score")
        else:
            print("🔴 FAILED - Adjust risk or indicator logic")
    else:
        print("No trades triggered. Confluence filter might be too strict.")

if __name__ == "__main__":
    run_confluence_backtest()