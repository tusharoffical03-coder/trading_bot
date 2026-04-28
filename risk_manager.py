import pandas as pd

class RiskManager:
    def __init__(self, capital=1000, max_risk_pct=2.0, max_daily_trades=3):
        self.capital = capital
        self.max_risk_pct = max_risk_pct
        self.max_daily_trades = max_daily_trades

    def can_trade(self, today_trade_count):
        """Check if we hit daily trade limit"""
        if today_trade_count >= self.max_daily_trades:
            print(f"[RISK] Daily trade limit reached ({self.max_daily_trades}). Skipping.")
            return False
        return True

    def calculate_position_size(self, entry, sl, confidence_score):
        """
        Position sizing based on AI confidence score.
        confidence_score is 0-100.
        """
        # Determine risk %
        if confidence_score >= 80:
            risk_pct = self.max_risk_pct
        elif 65 <= confidence_score < 80:
            risk_pct = self.max_risk_pct / 2.0  # Half risk
        else:
            print(f"[RISK] Confidence {confidence_score:.1f}% too low (< 65%). Skipping.")
            return 0.0, 0.0

        risk_amount = self.capital * (risk_pct / 100.0)
        
        # SL distance %
        sl_dist_pct = abs(entry - sl) / entry
        
        if sl_dist_pct == 0:
            return 0.0, 0.0
            
        # Size = Risk Amount / SL Distance
        position_size_usd = risk_amount / sl_dist_pct
        
        # Cap at some leverage factor if needed, assuming spot or 1-2x leverage max here
        # E.g. limit to 2x capital
        if position_size_usd > self.capital * 2:
            position_size_usd = self.capital * 2
            
        position_size_btc = position_size_usd / entry
        
        return position_size_btc, risk_pct
