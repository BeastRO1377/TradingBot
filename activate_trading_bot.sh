#!/bin/bash
# activate_trading_bot.sh - Script to activate Python 3.11.12 environment for TradingBot

echo "Activating Python 3.11.12 environment for TradingBot..."
source venv/bin/activate
echo "Environment activated!"
echo "Python version: $(python --version)"
echo "Virtual environment: $(which python)"
echo ""
echo "You can now run your TradingBot with:"
echo "  python main.py"
echo ""
echo "To deactivate the environment, run: deactivate"

