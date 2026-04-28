# ☁️ PythonAnywhere 24/7 Deployment Guide

PythonAnywhere cloud par bot ko deploy karna sabse aasan hai. Neeche diye gaye steps follow karein:

## Step 1: Account Banayein
1. [PythonAnywhere.com](https://www.pythonanywhere.com/) par jayen aur ek free account banayen.
2. Login karne ke baad apne **Dashboard** par jayen.

## Step 2: Files Upload Karein
1. Dashboard par **Files** tab par click karein.
2. Ek naya folder banayen (e.g., `btc_agent`).
3. Apne laptop (c:\btc_agent\) se yeh sabhi files upload karein:
   - `live_agent.py`
   - `data_fetcher.py`
   - `indicators.py`
   - `confluence.py`
   - `groq_anilyzer.py`
   - `bedrock_analyzer.py`
   - `model.py`
   - `risk_manager.py`
   - `trade_logger.py`
   - `Requirement.txt`
   - `.env`
   - `scaler.pkl`, `price_scaler.pkl`, `xgb.pkl`, `lstm.pt`

## Step 3: Dependencies Install Karein
1. Dashboard par **Consoles** tab me jaakar **Bash** console open karein.
2. Niche di gayi command run karein (folder me jane ke liye):
   ```bash
   cd btc_agent
   ```
3. Libraries install karne ke liye yeh run karein:
   ```bash
   pip3 install -r Requirement.txt --user
   ```

## Step 4: Bot Start Karein (24/7)

PythonAnywhere par script ko 24/7 chalane ke do tarike hain:

### Option A: Always-on Task (Recommended - Paid Plan)
Agar aapka PythonAnywhere ka paid plan ($5/mo) hai:
1. Dashboard me **Tasks** tab par jayen.
2. **Always-on tasks** section me, file ka path dalein: `/home/aapka_username/btc_agent/live_agent.py`
3. Usme `python` version 3.10 select karke start kar dein. Yeh 24/7 chalta rahega.

### Option B: Bash Console (Free Plan Hack)
Kyunki Free plan par 15-min scheduler nahi hota, aap Bash console me background process ki tarah chala sakte hain:
1. Bash console me jaakar run karein:
   ```bash
   nohup python3 live_agent.py &
   ```
2. Yeh background me chalna shuru ho jayega. (Note: Free tier me PythonAnywhere kabhi kabhi server restart karta hai toh aapko shayad har kuch din me ek baar ise dubara run karna pade).
