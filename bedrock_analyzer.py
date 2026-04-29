import os
import json
import boto3
from dotenv import load_dotenv

load_dotenv()

# Initialize Bedrock client
bedrock = None
bedrock_error = None

try:
    aws_key = os.getenv('AWS_ACCESS_KEY_ID', '').strip()
    aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY', '').strip()
    aws_region = os.getenv('AWS_REGION', 'us-east-1').strip()
    
    # Only attempt to initialize if credentials are provided
    if aws_key and aws_secret:
        bedrock = boto3.client(
            service_name='bedrock-runtime',
            region_name=aws_region,
            aws_access_key_id=aws_key,
            aws_secret_access_key=aws_secret
        )
    else:
        bedrock_error = "AWS credentials not provided (AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY missing)"
        print(f"[WARN] {bedrock_error}")
        
except Exception as e:
    bedrock_error = str(e)
    print(f"[ERROR] Failed to initialize Bedrock client: {bedrock_error}")

MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"

def analyze_trade_context_bedrock(market_data: dict, headlines: list):
    """
    Sends full technical confluence + news to Claude via AWS Bedrock.
    """
    if not bedrock:
        return _fallback_error(bedrock_error or "Bedrock client not initialized.")

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

Return ONLY valid JSON in this exact format (do not wrap in markdown tags):
{{
  "decision": "BUY", "SELL", or "SKIP",
  "confidence": <float 0-100>,
  "reasoning": "<1-2 short sentences explaining why>",
  "news_sentiment": <float -1 to 1>,
  "suggested_sl_distance_pct": <float e.g., 0.015 for 1.5%>,
  "suggested_tp_distance_pct": <float e.g., 0.03 for 3%>
}}
"""

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 400,
        "temperature": 0.2,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    })

    try:
        response = bedrock.invoke_model(
            body=body,
            modelId=MODEL_ID,
            accept='application/json',
            contentType='application/json'
        )
        
        response_body = json.loads(response.get('body').read())
        content_text = response_body.get('content')[0].get('text')
        
        # Ensure we just parse the JSON, handling any potential markdown block wrappers
        if "```json" in content_text:
            content_text = content_text.split("```json")[1].split("```")[0].strip()
        elif "```" in content_text:
            content_text = content_text.split("```")[1].split("```")[0].strip()
            
        return json.loads(content_text)
        
    except Exception as e:
        print(f"[ERROR] AWS Bedrock Analysis Failed: {e}")
        return _fallback_error(str(e))

def _fallback_error(error_msg):
    return {
        "decision": "SKIP",
        "confidence": 0,
        "reasoning": f"AWS Bedrock Error: {error_msg}",
        "news_sentiment": 0,
        "suggested_sl_distance_pct": 0.015,
        "suggested_tp_distance_pct": 0.03
    }
