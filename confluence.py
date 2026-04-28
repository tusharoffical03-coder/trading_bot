import pandas as pd
import numpy as np

# ═══════════════════════════════════════════════════════════════
# 6-Signal Confluence Engine
# ═══════════════════════════════════════════════════════════════

def check_htf_trend(df_4h):
    """
    Signal 1: HTF Trend Direction (4H)
    Bullish: EMA 9 > EMA 21 > EMA 50
    """
    if len(df_4h) < 1: return "Neutral"
    last = df_4h.iloc[-1]
    
    if pd.isna(last.get('ema50')):
        return "Neutral"
        
    if last['ema9'] > last['ema21'] and last['ema21'] > last['ema50']:
        return "Bullish"
    elif last['ema9'] < last['ema21'] and last['ema21'] < last['ema50']:
        return "Bearish"
    return "Neutral"

def check_sr_proximity(current_price, supports, resistances, threshold_pct=0.005):
    """
    Signal 2: S/R Zone Touched
    Is price within 0.5% of a key S/R level?
    """
    thresh = current_price * threshold_pct
    
    for s in supports[-5:]: # Check recent 5 supports
        if abs(current_price - s) <= thresh:
            return "Bullish" # At support
            
    for r in resistances[-5:]: # Check recent 5 resistances
        if abs(current_price - r) <= thresh:
            return "Bearish" # At resistance
            
    return "Neutral"

def check_momentum(row):
    """
    Signal 3: Momentum Confirmation (RSI + MACD)
    Bullish: RSI 40-60 (pullback, not overbought) + MACD > Signal
    Bearish: RSI 40-60 (pullback, not oversold) + MACD < Signal
    """
    rsi = row['rsi']
    macd = row['macd']
    macd_sig = row['macd_signal']
    
    if 40 <= rsi <= 60 and macd > macd_sig:
        return "Bullish"
    elif 40 <= rsi <= 60 and macd < macd_sig:
        return "Bearish"
    return "Neutral"

def check_volume(row):
    """
    Signal 4: Volume Confirmation
    Volume spike > 1.5x of 20-MA
    """
    if row.get('vol_ratio', 0) >= 1.5:
        # If bullish candle + high vol -> Bullish, else Bearish
        return "Bullish" if row['close'] > row['open'] else "Bearish"
    return "Neutral"

def check_candle_pattern(row):
    """
    Signal 5: Candlestick Pattern
    """
    if row.get('is_bull_pinbar') or row.get('is_bull_engulfing'):
        return "Bullish"
    if row.get('is_bear_pinbar') or row.get('is_bear_engulfing'):
        return "Bearish"
    return "Neutral"

def calculate_confluence_score(df_15m, df_1h, df_4h, current_price, supports, resistances, market_structure):
    """
    Calculate the 6-signal confluence score.
    Returns: score (int 0-6), direction ('LONG', 'SHORT', 'NEUTRAL'), details (dict)
    """
    if df_15m.empty or df_1h.empty or df_4h.empty:
        return 0, "NEUTRAL", {}
        
    last_15m = df_15m.iloc[-1]
    
    s1 = check_htf_trend(df_4h)
    s2 = check_sr_proximity(current_price, supports, resistances)
    s3 = check_momentum(last_15m)
    s4 = check_volume(last_15m)
    s5 = check_candle_pattern(last_15m)
    s6 = market_structure # Bullish/Bearish/Ranging from 1H
    
    signals = {
        "HTF_Trend": s1,
        "SR_Zone": s2,
        "Momentum": s3,
        "Volume": s4,
        "Candle": s5,
        "Market_Structure": s6
    }
    
    bull_score = sum(1 for v in signals.values() if v == "Bullish")
    bear_score = sum(1 for v in signals.values() if v == "Bearish")
    
    if bull_score >= 4:
        return bull_score, "LONG", signals
    elif bear_score >= 4:
        return bear_score, "SHORT", signals
        
    # If neither is >= 4, check which one is higher, or default to Neutral
    if bull_score > bear_score:
        return bull_score, "NEUTRAL", signals
    else:
        return bear_score, "NEUTRAL", signals
