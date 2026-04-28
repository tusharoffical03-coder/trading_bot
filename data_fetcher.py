import pandas as pd
import requests
from datetime import datetime
import time

# ═══════════════════════════════════════════════════════════════
# Multi-Timeframe BTC Data Fetcher
# Supports: 15m, 1H, 4H candles from Binance Public API
# ═══════════════════════════════════════════════════════════════

def fetch_btc_candles(interval='1h', limit=1000, symbol='BTCUSDT'):
    """Fetch BTC candles from Binance public API — NO KEY NEEDED."""
    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    end_time = None
    remaining = limit
    while remaining > 0:
        fetch_amount = min(1000, remaining)
        params = {"symbol": symbol, "interval": interval, "limit": fetch_amount}
        if end_time:
            params['endTime'] = end_time
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code != 200:
                print(f"[ERROR] Binance API error ({interval}): {r.text}")
                break
            data = r.json()
            if not data:
                break
            all_data = data + all_data
            remaining -= len(data)
            end_time = data[0][0] - 1
            if remaining > 0:
                time.sleep(0.1)
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Network error fetching {interval}: {e}")
            break
    if not all_data:
        return pd.DataFrame()
    df = pd.DataFrame(all_data, columns=[
        'time','open','high','low','close','volume',
        'close_time','qav','trades','tbbav','tbqav','ignore'
    ])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    for c in ['open','high','low','close','volume']:
        df[c] = df[c].astype(float)
    df = df.drop_duplicates(subset=['time']).sort_values('time').reset_index(drop=True)
    return df[['time','open','high','low','close','volume']]

def fetch_btc_15m(limit=1000):
    """Fetch 15-minute candles for entry timing."""
    return fetch_btc_candles(interval='15m', limit=limit)

def fetch_btc_1h(limit=1000):
    """Fetch 1-hour candles — primary analysis timeframe."""
    return fetch_btc_candles(interval='1h', limit=limit)

def fetch_btc_4h(limit=500):
    """Fetch 4-hour candles for HTF trend confirmation."""
    return fetch_btc_candles(interval='4h', limit=limit)

def fetch_btc_daily(limit=200):
    """Fetch daily candles for major S/R detection."""
    return fetch_btc_candles(interval='1d', limit=limit)

def fetch_multi_timeframe():
    """Fetch all timeframes. Returns dict with '15m','1h','4h','1d' DataFrames."""
    print("[DATA] Fetching multi-timeframe data...")
    data = {}
    data['15m'] = fetch_btc_15m(limit=500)
    print(f"  15m: {len(data['15m'])} candles")
    data['1h'] = fetch_btc_1h(limit=1000)
    print(f"  1h:  {len(data['1h'])} candles")
    data['4h'] = fetch_btc_4h(limit=500)
    print(f"  4h:  {len(data['4h'])} candles")
    data['1d'] = fetch_btc_daily(limit=200)
    print(f"  1d:  {len(data['1d'])} candles")
    return data

def fetch_fear_greed_index():
    """Fetch Crypto Fear & Greed Index from alternative.me API."""
    try:
        url = "https://api.alternative.me/fng/?limit=1&format=json"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()['data'][0]
            return {
                'value': int(data['value']),
                'classification': data['value_classification'],
                'timestamp': datetime.fromtimestamp(int(data['timestamp']))
            }
    except Exception as e:
        print(f"[WARN] Fear & Greed API error: {e}")
    return {'value': 50, 'classification': 'Neutral', 'timestamp': datetime.now()}

if __name__ == "__main__":
    mtf = fetch_multi_timeframe()
    for tf, df in mtf.items():
        if not df.empty:
            print(f"{tf} last: {df.iloc[-1]['time']} | ${df.iloc[-1]['close']:.2f}")
    fg = fetch_fear_greed_index()
    print(f"Fear & Greed: {fg['value']} ({fg['classification']})")