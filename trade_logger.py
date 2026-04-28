import os
import csv
from datetime import datetime
from supabase import create_client, Client

class TradeLogger:
    def __init__(self, log_file="trades.csv"):
        self.log_file = log_file
        self._init_csv()
        
        # Init Supabase
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        if supabase_url and supabase_key and supabase_url != "your_supabase_project_url_here":
            self.supabase: Client = create_client(supabase_url, supabase_key)
        else:
            self.supabase = None

    def _init_csv(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Time", "Symbol", "Direction", "Entry", "SL", "TP",
                    "Confluence_Score", "AI_Confidence", "AI_Reasoning", 
                    "Position_Risk_Pct", "Status", "PnL"
                ])

    def log_signal(self, symbol, direction, entry, sl, tp, conf_score, ai_conf, ai_reason, risk_pct):
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Log to local CSV
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp_str,
                symbol, direction, round(entry, 2), round(sl, 2), round(tp, 2),
                f"{conf_score}/6", f"{ai_conf:.1f}%", ai_reason,
                f"{risk_pct}%", "OPEN", 0.0
            ])
            
        # Log to Cloud Supabase
        if self.supabase:
            try:
                data, count = self.supabase.table('trades').insert({
                    "time": timestamp_str,
                    "symbol": symbol,
                    "direction": direction,
                    "entry_price": round(entry, 2),
                    "stop_loss": round(sl, 2),
                    "take_profit": round(tp, 2),
                    "confluence_score": conf_score,
                    "ai_confidence": float(ai_conf),
                    "ai_reasoning": ai_reason,
                    "status": "OPEN",
                    "pnl": 0.0
                }).execute()
                print("[OK] Trade synced to Supabase Cloud Dashboard.")
            except Exception as e:
                print(f"[WARN] Failed to sync to Supabase: {e}")
            
    def update_outcomes(self, current_price):
        """Check all 'OPEN' trades and see if they hit TP or SL"""
        
        # 1. Update Supabase
        if self.supabase:
            try:
                # Fetch open trades from cloud
                res = self.supabase.table('trades').select('*').eq('status', 'OPEN').execute()
                open_trades = res.data
                
                for trade in open_trades:
                    direction = trade['direction']
                    tp = float(trade['take_profit'])
                    sl = float(trade['stop_loss'])
                    entry = float(trade['entry_price'])
                    
                    status = "OPEN"
                    pnl = 0.0
                    
                    if direction == "LONG":
                        if current_price >= tp:
                            status = "WIN"
                            pnl = (tp - entry) / entry * 100
                        elif current_price <= sl:
                            status = "LOSS"
                            pnl = (sl - entry) / entry * 100
                    else: # SHORT
                        if current_price <= tp:
                            status = "WIN"
                            pnl = (entry - tp) / entry * 100
                        elif current_price >= sl:
                            status = "LOSS"
                            pnl = (entry - sl) / entry * 100
                            
                    if status != "OPEN":
                        self.supabase.table('trades').update({
                            "status": status,
                            "pnl": round(pnl, 2)
                        }).eq('id', trade['id']).execute()
                        print(f"[EVENT] Trade {trade['id']} closed as {status} (PnL: {pnl:.2f}%)")
                        
            except Exception as e:
                print(f"[WARN] Supabase update failed: {e}")

        # 2. Update Local CSV (Simplified: just log to console for now as Supabase is the primary dashboard source)
        # In a real app, we'd rewrite the CSV, but Supabase is what the user sees on Vercel.

    def get_daily_trade_count(self):
        """Count how many trades were taken today."""
        if not os.path.exists(self.log_file): return 0
        
        today = datetime.now().strftime("%Y-%m-%d")
        count = 0
        with open(self.log_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['Time'].startswith(today):
                    count += 1
        return count
