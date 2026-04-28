import os
import requests
from dotenv import load_dotenv

load_dotenv()

def send_test_notification():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    print(f"Testing Telegram with Token: {token[:10]}... and Chat ID: {chat_id}")
    
    if not token or not chat_id or token == "your_bot_token_here":
        print("Error: Telegram credentials missing in .env")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    message = "✅ <b>BTC Trading Agent Connected!</b>\n\nAapka phone bot se successfully link ho gaya hai. Ab saare trading signals isi tarah yahan aayenge. 🚀"
    
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML"
    }
    
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code == 200:
            print("Successfully sent test notification to Telegram!")
        else:
            print(f"Failed to send notification. Error: {r.text}")
    except Exception as e:
        print(f"Exception occurred: {e}")

if __name__ == "__main__":
    send_test_notification()
