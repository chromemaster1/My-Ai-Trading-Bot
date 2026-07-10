# AI Trading Bot

A Flask-based trading dashboard with paper/live trading modes, live exchange connectivity for Binance and Bybit, and production-friendly environment configuration.

## Quick start

1. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Copy the example environment file and update it:
   ```bash
   cp .env.example .env
   ```
3. Start the bot:
   ```bash
   python trading_bot_main.py
   ```
4. Open http://localhost:5000 and sign in with the admin credentials from your environment variables.

## Production deployment

The bot is compatible with Gunicorn:

```bash
gunicorn --bind 0.0.0.0:$PORT trading_bot_main:app
```

Use environment variables or a hosted secret store for your API keys. The bot stores the local state file in `.trading_bot_state.json` and ignores it in Git.

## Features

- Paper and live trading toggle
- Binance and Bybit exchange connection setup
- Home dashboard balance and exchange status
- Authenticated admin settings for account linking and risk controls
- Health endpoint at /healthz
