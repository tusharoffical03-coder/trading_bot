import pandas as pd
import numpy as np
import ta

def add_indicators(df):
    """
    Enhanced indicator suite with basic ML features and new confluence signals.
    """
    df = df.copy()
    
    # 1. Base Momentum & Trend
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], 14).rsi()
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    
    bb = ta.volatility.BollingerBands(df['close'], 20, 2)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_pct'] = bb.bollinger_pband()
    
    # 2. Moving Averages (Old + New)
    df['ema9'] = ta.trend.EMAIndicator(df['close'], 9).ema_indicator()
    df['ema21'] = ta.trend.EMAIndicator(df['close'], 21).ema_indicator()
    df['ema20'] = ta.trend.EMAIndicator(df['close'], 20).ema_indicator()
    df['ema50'] = ta.trend.EMAIndicator(df['close'], 50).ema_indicator()
    
    # 3. Volatility & Volume
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], 14).average_true_range()
    df['vol_ma'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma'].replace(0, np.nan)
    
    df['returns'] = df['close'].pct_change()
    df['vp_trend'] = df['volume'] * np.sign(df['returns'])
    
    # 4. Advanced ML Features (Existing)
    df['stoch'] = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], 14).stoch()
    df['willr'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close'], 14).williams_r()
    df['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close'], 20).cci()
    df['mfi'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume'], 14).money_flow_index()
    
    df['dist_ema20'] = (df['close'] - df['ema20']) / df['ema20']
    df['dist_ema50'] = (df['close'] - df['ema50']) / df['ema50']
    
    # 5. Candlestick Patterns
    df['body'] = abs(df['close'] - df['open'])
    df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
    df['range'] = df['high'] - df['low']
    
    # Pin bar / Hammer (Lower shadow > 2x body, upper shadow small)
    df['is_bull_pinbar'] = (df['lower_shadow'] >= 2 * df['body']) & (df['upper_shadow'] <= 0.5 * df['body']) & (df['close'] > df['open'])
    # Shooting star (Upper shadow > 2x body, lower shadow small)
    df['is_bear_pinbar'] = (df['upper_shadow'] >= 2 * df['body']) & (df['lower_shadow'] <= 0.5 * df['body']) & (df['close'] < df['open'])
    
    # Engulfing (Current body engulfs previous body)
    prev_body = df['body'].shift(1)
    prev_close = df['close'].shift(1)
    prev_open = df['open'].shift(1)
    
    df['is_bull_engulfing'] = (df['close'] > df['open']) & (prev_close < prev_open) & \
                              (df['close'] > prev_open) & (df['open'] < prev_close)
    df['is_bear_engulfing'] = (df['close'] < df['open']) & (prev_close > prev_open) & \
                              (df['close'] < prev_open) & (df['open'] > prev_close)

    df = df.dropna().reset_index(drop=True)
    return df

def find_sr_zones(df, window=20):
    """
    Identify Support/Resistance zones using local highs/lows.
    """
    df = df.copy()
    df['is_support'] = df['low'] == df['low'].rolling(window=window, center=True).min()
    df['is_resistance'] = df['high'] == df['high'].rolling(window=window, center=True).max()
    
    supports = df[df['is_support']]['low'].values
    resistances = df[df['is_resistance']]['high'].values
    
    return supports, resistances

def detect_market_structure(df, window=10):
    """
    Returns 'Bullish', 'Bearish', or 'Ranging' based on Higher Highs/Lows.
    """
    recent = df.tail(window * 3).copy()
    recent['local_high'] = recent['high'] == recent['high'].rolling(window=window, center=True).max()
    recent['local_low'] = recent['low'] == recent['low'].rolling(window=window, center=True).min()
    
    highs = recent[recent['local_high']]['high'].values
    lows = recent[recent['local_low']]['low'].values
    
    if len(highs) >= 2 and len(lows) >= 2:
        if highs[-1] > highs[-2] and lows[-1] > lows[-2]:
            return "Bullish" # HH and HL
        elif highs[-1] < highs[-2] and lows[-1] < lows[-2]:
            return "Bearish" # LH and LL
            
    return "Ranging"