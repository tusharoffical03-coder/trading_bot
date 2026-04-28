import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "llama-3.3-70b-versatile"  # Fast, powerful model

def analyze_trade_context(market_data: dict, headlines: list):
    """
    Enhanced prompt sending full technical confluence + news to Groq.
    """
    news_text = "\n".join([f"- {h}" for h in headlines[:10]])
    
    prompt = f"""You are an expert BTC quant trader. Analyze this technical confluence data and recent news to give a clear BUY / SELL / SKIP decision.

MARKET DATA:
- Current BTC Price: ${market_data['current_price']:.2f}
- Confluence Score: {market_data['confluence_score']}/6
- ML Ensemble Direction: {market_data['ml_direction']} (Prob: {market_data['ml_prob']:.2f})
- 4H Trend: {market_data['signals'].get('HTF_Trend')}
- S/R Proximity: {market_data['signals'].get('SR_Zone')}
- Momentum (RSI+MACD): {market_data['signals'].get('Momentum')}
- Volume Spike: {market_data['signals'].get('Volume')}
- Candle Pattern: {market_data['signals'].get('Candle')}
- Market Structure: {market_data['signals'].get('Market_Structure')}
- Fear & Greed Index: {market_data['fear_greed']}

RECENT NEWS:
{news_text}

Return ONLY valid JSON in this exact format:
{{
  "decision": "BUY", "SELL", or "SKIP",
  "confidence": <float 0-100>,
  "reasoning": "<1-2 short sentences explaining why>",
  "news_sentiment": <float -1 to 1>,
  "suggested_sl_distance_pct": <float e.g., 0.015 for 1.5%>,
  "suggested_tp_distance_pct": <float e.g., 0.03 for 3%>
}}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a professional crypto analyst. Output only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=400,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"[ERROR] Groq Analysis Failed: {e}")
        return {
            "decision": "SKIP",
            "confidence": 0,
            "reasoning": f"API Error: {e}",
            "news_sentiment": 0,
            "suggested_sl_distance_pct": 0.015,
            "suggested_tp_distance_pct": 0.03
        }

if __name__ == "__main__":
    # Test data
    sample_data = {
        'current_price': 65000, 'confluence_score': 5, 'ml_direction': 'LONG', 'ml_prob': 0.75,
        'signals': {'HTF_Trend': 'Bullish', 'SR_Zone': 'Neutral', 'Momentum': 'Bullish', 
                    'Volume': 'Bullish', 'Candle': 'Bullish', 'Market_Structure': 'Bullish'},
        'fear_greed': 75
    }
    sample_news = ["Bitcoin breaks $65k resistance!", "ETF inflows hit record high."]
    
    print(json.dumps(analyze_trade_context(sample_data, sample_news), indent=2))