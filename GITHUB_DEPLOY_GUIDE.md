# 🐙 GitHub Actions 24/7 Deployment Guide

Is guide ki madad se aapka BTC Trading Bot GitHub ke servers par **24/7 Free** chalega. Aapka laptop off ho, tab bhi ye har 15 minute mein market scan karke aapko Telegram bhejta rahega.

## Step 1: GitHub Repository Banayein
1. [GitHub.com](https://github.com/) par login karein.
2. Naya repository banayein: **New Repository**.
3. Repository Name: `btc_agent`
4. **Important:** Isse **PRIVATE** rakhein taaki aapka code aur logic safe rahe.
5. "Create repository" par click karein.

## Step 2: Files Upload Karein
1. Apne laptop se niche di gayi saari files upload karein:
   - `live_agent.py`, `data_fetcher.py`, `indicators.py`, `confluence.py`
   - `groq_anilyzer.py`, `bedrock_analyzer.py`, `news_analyzer.py`
   - `model.py`, `risk_manager.py`, `trade_logger.py`
   - `Requirement.txt`
   - `.github/` folder (jo maine abhi banaya hai)
   - `.pkl` aur `.pt` files (ML Models)
2. **⚠️ ALERT:** `.env` file kabhi bhi upload mat karna.

## Step 3: API Keys Setup Karein (Secrets)
Kyuki humne `.env` upload nahi kiya, hume GitHub ko keys batani hongi:
1. GitHub repository mein upar **Settings** par click karein.
2. Left menu mein **Secrets and variables > Actions** par jayen.
3. **New repository secret** par click karein aur niche di gayi saari keys ek-ek karke add karein (jaisa aapki `.env` mein hai):
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`
   - `SUPABASE_URL`
   - `SUPABASE_KEY`
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
   - `AWS_REGION`
   - `AI_PROVIDER` (Value: `bedrock`)
   - `CAPITAL` (Value: `1000`)
   - `MAX_RISK_PCT` (Value: `2.0`)
   - `MAX_DAILY_TRADES` (Value: `100`)

## Step 4: Bot Start Ho Gaya!
1. GitHub automatic har 15 minute mein script run karna shuru kar dega.
2. Aap check karne ke liye repository mein **Actions** tab par ja sakte hain.
3. Jab bhi signal aayega, aapko Telegram mil jayega!

**Fayda:** Ab aap apna laptop band kar sakte hain, bot cloud par hamesha chalta rahega.
