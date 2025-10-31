#!/bin/bash

# Create migration directory
mkdir -p migration_files

# Copy main bot files
cp MultiuserBot_V16.py migration_files/
cp telegram_fsm_v12.py migration_files/
cp requirements.txt migration_files/
cp requirements_conda.txt migration_files/

# Copy configuration files
cp user_state*.json migration_files/
cp keys_TESTNET*.env migration_files/
cp com.multiuser.bot.plist migration_files/

# Copy ML models and data
mkdir -p migration_files/models
cp -r models/* migration_files/models/
cp trading_model_final.pkl migration_files/
cp trades_for_training.csv migration_files/
cp golden_setup_snapshots.csv migration_files/

# Copy important logs and data
cp open_positions.json migration_files/
cp wallet_state.json migration_files/
cp trades_unified.csv migration_files/
cp trade_log_MUWS.log migration_files/

# Copy training scripts
cp train_lightgbm.py migration_files/

# Create archive
tar -czf trading_bot_migration.tar.gz migration_files/

# Cleanup
rm -rf migration_files

echo "Migration package created: trading_bot_migration.tar.gz" 