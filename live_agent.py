import torch, joblib, schedule, time, os, requests, sys
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

# Import our new multi-layer modules
from data_fetcher import fetch_multi_timeframe, fetch_fear_greed_index
from indicators import add_indicators, find_sr_zones, detect_market_structure
from confluence import calculate_confluence_score
from groq_anilyzer import analyze_trade_context as analyze_trade_context_groq
from bedrock_analyzer import analyze_trade_context_bedrock
from news_analyzer import get_news_sentiment
from risk_manager import RiskManager
from trade_logger import TradeLogger
from model import TimeSeriesTransformer, FEATURES

# Create global logger for use in monitoring
logger = TradeLogger()

load_dotenv()

def validate_and_load_env():
    """Validate and set defaults for critical environment variables"""
    defaults = {
        "CAPITAL": "1000",
        "MAX_RISK_PCT": "2.0",
        "MAX_DAILY_TRADES": "100",
        "AI_PROVIDER": "groq"
    }
    
    for var, default_val in defaults.items():
        value = os.getenv(var, "").strip()
        if not value:
            os.environ[var] = default_val
            print(f"[WARN] {var} was empty or missing, using default: {default_val}")

def run_live_agent():
    print(f"\n[{datetime.now()}] >>> Starting Multi-Layer Confluence Analysis <<<")
    
    # Validate environment variables first
    validate_and_load_env()
    
    # Check outcomes of previous trades first
    check_open_trades()
    
    # 1. Init helpers
    try:
        capital = float(os.getenv("CAPITAL", "1000"))
        max_risk_pct = float(os.getenv("MAX_RISK_PCT", "2.0"))
        max_daily_trades = int(os.getenv("MAX_DAILY_TRADES", "100"))
    except (ValueError, TypeError) as e:
        print(f"[ERROR] Invalid configuration values: {e}")
        return
    
    risk_mgr = RiskManager(
        capital=capital,
        max_risk_pct=max_risk_pct,
        max_daily_trades=max_daily_trades
    )
    
    if not risk_mgr.can_trade(logger.get_daily_trade_count()):
        return

    # 2. Fetch Multi-TF Data
    try:
        mtf = fetch_multi_timeframe()
        if mtf['15m'].empty or mtf['1h'].empty or mtf['4h'].empty:
            print("[ERROR] Failed to fetch required timeframes. Skipping.")
            return
    except Exception as e:
        print(f"[ERROR] Failed to fetch market data: {e}")
        return

    try:
        fg_index = fetch_fear_greed_index()
    except Exception as e:
        print(f"[WARN] Failed to fetch fear/greed index: {e}")
        fg_index = {'value': 50}  # Neutral default

    # 3. Apply Indicators
    df_15m = add_indicators(mtf['15m'])
    df_1h = add_indicators(mtf['1h'])
    df_4h = add_indicators(mtf['4h'])
    df_1d = add_indicators(mtf['1d'])

    current_price = df_15m['close'].iloc[-1]
    
    # 4. S/R & Market Structure (from 1D & 1H)
    supports, resistances = find_sr_zones(df_1d, window=10)
    structure = detect_market_structure(df_1h, window=10)

    # 5. Confluence Scoring
    score, conf_dir, signals = calculate_confluence_score(
        df_15m, df_1h, df_4h, current_price, supports, resistances, structure
    )
    
    print(f"\n--- LAYER 1: CONFLUENCE Filter ---")
    print(f"Market Structure: {structure}")
    for k, v in signals.items():
        print(f"{k}: {v}")
    print(f"TOTAL SCORE: {score}/6 | Direction: {conf_dir}")

    # Early exit if confluence is weak
    if score < 4 and conf_dir != "NEUTRAL":
        print(f"[SKIP] Confluence score {score}/6 is below threshold (4/6).")
        return
    elif conf_dir == "NEUTRAL":
        print("[SKIP] No clear directional confluence.")
        return




    # 6. ML Ensemble Prediction (Transformer + XGBoost)
    print("\n--- LAYER 2: ML Ensemble ---")
    try:
        scaler = joblib.load('scaler.pkl')
        xgb = joblib.load('xgb.pkl')
        model = TimeSeriesTransformer(len(FEATURES))
        model.load_state_dict(torch.load('lstm.pt', map_location='cpu'))
        model.eval()

        # We use 1H data for ML models as trained
        seq = df_1h[FEATURES].values[-12:]
        seq_s = scaler.transform(seq).reshape(1, 12, -1)
        
        xgb_prob = xgb.predict_proba(seq.reshape(1, -1))[0][1]
        with torch.no_grad():
            _, trans_prob = model(torch.tensor(seq_s, dtype=torch.float32))
            trans_prob = trans_prob.item()
            
        ml_prob = (xgb_prob + trans_prob) / 2
        ml_dir = "LONG" if ml_prob > 0.5 else "SHORT"
        
        print(f"XGB: {xgb_prob:.2f} | Trans: {trans_prob:.2f} | Avg: {ml_prob:.2f}")
        print(f"ML Direction: {ml_dir}")
        
    except Exception as e:
        print(f"[ERROR] ML Models failed: {e}")
        return

    # Check ML Alignment with Confluence
    if conf_dir != ml_dir:
        print(f"[SKIP] Confluence ({conf_dir}) disagrees with ML ({ml_dir}).")
        return



    # 7. AI Validation (Groq or Bedrock)
    ai_provider = os.getenv("AI_PROVIDER", "groq").lower().strip()
    print(f"\n--- LAYER 3: AI Validation ({ai_provider.upper()}) ---")
    
    try:
        _, headlines_raw = get_news_sentiment(limit=10)
        headlines = [h[0] for h in headlines_raw] if headlines_raw else []
    except Exception as e:
        print(f"[WARN] Failed to fetch news: {e}")
        headlines = []
    
    market_context = {
        'current_price': current_price,
        'confluence_score': score,
        'ml_direction': ml_dir,
        'ml_prob': ml_prob,
        'signals': signals,
        'fear_greed': fg_index.get('value', 50)
    }
    
    if ai_provider == "bedrock":
        ai_response = analyze_trade_context_bedrock(market_context, headlines)
    else:
        ai_response = analyze_trade_context_groq(market_context, headlines)
        
    ai_dec = ai_response.get('decision', 'SKIP')
    ai_conf = ai_response.get('confidence', 0)

    
    print(f"AI Decision: {ai_dec} | Confidence: {ai_conf}%")
    print(f"Reason: {ai_response.get('reasoning')}")

    # Final Decision Gate
    mapped_ai_dec = "LONG" if ai_dec == "BUY" else "SHORT" if ai_dec == "SELL" else "SKIP"
    
    if mapped_ai_dec != conf_dir or ai_conf < 65:
        print(f"[SKIP] AI Validation failed. (AI: {ai_dec}, Conf: {ai_conf}%)")
        return


        
    # 8. Risk Management & Position Sizing
    sl_pct = ai_response.get('suggested_sl_distance_pct', 0.015)
    tp_pct = ai_response.get('suggested_tp_distance_pct', 0.03)
    
    sl = current_price * (1 - sl_pct) if conf_dir == "LONG" else current_price * (1 + sl_pct)
    tp = current_price * (1 + tp_pct) if conf_dir == "LONG" else current_price * (1 - tp_pct)
    
    pos_btc, risk_pct = risk_mgr.calculate_position_size(current_price, sl, ai_conf)
    
    if pos_btc <= 0:
        print("[SKIP] Risk manager rejected size.")
        return

    # 9. SIGNAL FIRED
    print(f"\nSIGNAL FIRED: {conf_dir}")
    print(f"Entry: ${current_price:.2f}")
    print(f"SL: ${sl:.2f} | TP: ${tp:.2f}")
    print(f"Size: {pos_btc:.4f} BTC (Risking {risk_pct}%)")


    logger.log_signal(
        symbol="BTCUSDT", direction=conf_dir, entry=current_price, 
        sl=sl, tp=tp, conf_score=score, ai_conf=ai_conf, 
        ai_reason=ai_response.get('reasoning', ''), risk_pct=risk_pct
    )
    
    # Notify n8n Webhook
    webhook_url = os.getenv("N8N_WEBHOOK_URL", "").strip()
    if webhook_url:
        try:
            payload = {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": "BTCUSDT",
                "direction": conf_dir,
                "entry": current_price,
                "sl": sl, "tp": tp,
                "confluence": f"{score}/6",
                "ai_confidence": f"{ai_conf}%",
                "ai_reasoning": ai_response.get('reasoning', '')
            }
            requests.post(webhook_url, json=payload, timeout=5)
            print("[OK] Webhook sent!")
        except Exception as e:
            print(f"[WARN] Webhook failed: {e}")
            
    # Send Telegram Notification
    tg_msg = (
        f"🚨 <b>BTC SIGNAL FIRED</b> 🚨\n\n"
        f"<b>Direction:</b> {conf_dir}\n"
        f"<b>Entry:</b> ${current_price:.2f}\n"
        f"<b>Take Profit:</b> ${tp:.2f}\n"
        f"<b>Stop Loss:</b> ${sl:.2f}\n"
        f"<b>Size:</b> {pos_btc:.4f} BTC (Risk: {risk_pct}%)\n\n"
        f"<b>Confluence:</b> {score}/6\n"
        f"<b>Claude Confidence:</b> {ai_conf}%\n"
        f"<b>Reason:</b> {ai_response.get('reasoning', '')}"
    )
    send_telegram_message(tg_msg)

def send_telegram_message(message):
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if token and chat_id and token != "your_bot_token_here":
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
        try:
            requests.post(url, json=payload, timeout=5)
        except Exception as e:
            print(f"[WARN] Telegram message failed: {e}")

def check_open_trades():
    """Monitor open trades and update their status (Win/Loss)"""
    print("[MONITOR] Checking status of open trades...")
    
    # Get current BTC price for monitoring
    try:
        from data_fetcher import fetch_btc_1h
        df_now = fetch_btc_1h(limit=1)
        current_price = df_now['close'].iloc[-1]
    except:
        return

    # Update local CSV and Supabase
    logger.update_outcomes(current_price)

if __name__ == "__main__":
    if "--once" in sys.argv:
        run_live_agent()
    else:
        run_live_agent()
        # Schedule every 1 minute for faster reaction
        schedule.every(1).minutes.do(run_live_agent)
        print("Agent scheduler running (1-minute intervals)...")
        while True:
            schedule.run_pending()
            time.sleep(30)
