import requests
import asyncio
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware
import time
import json
from telegram import Bot
import csv
import os
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler

def setup_logging():
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, "DEX_bot.log")
    
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

# Конфигурация
TELEGRAM_TOKEN = 'AAF3awK3NRisz5dzCxfK2jMVC26W6D2DV-E'
TELEGRAM_CHAT_ID = '36972091'

# Пример корректных endpoint-ов BLAST (замените на свои настоящие):
INFURA_URL = 'https://eth-mainnet.blastapi.io/ce90ca43-0f44-4d49-9402-b41b81d72169'
BSC_NODE_URL = 'https://bsc-mainnet.blastapi.io/ce90ca43-0f44-4d49-9402-b41b81d72169'

UNISWAP_FACTORY_ADDRESS = '0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f'
PANCAKE_FACTORY_ADDRESS = '0xcA143Ce32Fe78f1f7019d7d551a6402fC5350c73'
LOG_FILE = "new_pairs_log.csv"

LOG_FIELDS = [
    'timestamp',
    'chain',
    'token_address',
    'pair_address',
    'token_symbol',
    'liquidity',
    'honeypot',
    'rugpull_risk',
    'social_activity',
    'total_score',
    'liquidity_score',
    'contract_score',
    'social_score',
    'volume_score',
    'holder_score'
]

telegram_bot = Bot(token=TELEGRAM_TOKEN)

def init_web3(provider_url, is_poa=False):
    # Если BLAST требует api-key в заголовках, добавляем request_kwargs
    w3 = Web3(Web3.HTTPProvider(provider_url, request_kwargs={
        "headers": {"x-api-key": "ty1UVRXbMmE4KwATAA0aazpjjRmd8-E31LNgYbNWfns"}
    }))
    if is_poa:
        w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
    return w3

factory_abi = json.loads("""
[
  {
    "constant": true,
    "inputs": [
      {"internalType": "address","name": "","type": "address"},
      {"internalType": "address","name": "","type": "address"}
    ],
    "name": "getPair",
    "outputs": [
      {"internalType": "address","name":"","type":"address"}
    ],
    "payable": false,
    "stateMutability":"view",
    "type":"function"
  },
  {
    "anonymous": false,
    "inputs": [
      {"indexed":true,"internalType":"address","name":"token0","type":"address"},
      {"indexed":true,"internalType":"address","name":"token1","type":"address"},
      {"indexed":false,"internalType":"address","name":"pair","type":"address"},
      {"indexed":false,"internalType":"uint256","name":"","type":"uint256"}
    ],
    "name":"PairCreated",
    "type":"event"
  }
]
""")

TOKEN_ABI = [
    {
        "constant": True,
        "inputs": [],
        "name": "symbol",
        "outputs": [{"name": "", "type": "string"}],
        "payable": False,
        "type": "function"
    }
]

# Заглушка: если у вас WETH/WBNB, можно сюда добавлять и другие «базовые»/«стабильные» токены
STABLE_TOKENS = {
    '0xC02aaa39b223FE8D0A0e5C4F27eAD9083C756Cc2',  # WETH на Ethereum
    '0xBB4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c',  # WBNB на BSC
    # USDT, USDC, BUSD, DAI — по ситуации
}

def check_honeypot(token_address, chain):
    """Запрос к API honeypot.is (пример). На практике может требовать API-key."""
    try:
        if chain == 'eth':
            url = f'https://api.honeypot.is/v2/IsHoneypot?address={token_address}'
        else:
            # 56 — chainId BSC
            url = f'https://api.honeypot.is/v2/IsHoneypot?address={token_address}&chainId=56'
        
        response = requests.get(url, timeout=10)
        data = response.json()
        if 'honeypotResult' in data:
            return data['honeypotResult']['isHoneypot']
        return False
    except Exception as e:
        logger.warning(f"Honeypot check error: {e}")
        # Логика: либо вернуть None, либо False
        return False

def check_rugpull(token_address, w3):
    """Упрощённая проверка на rugpull: проверяем, что кода нет или почти нет."""
    try:
        contract_code = w3.eth.get_code(token_address)
        # Считаем rugpull, если код короткий/отсутствует
        return len(contract_code) <= 2
    except:
        return False

def check_social_links(token_symbol):
    """Заглушка. Возвращаем всегда True или некий dict."""
    return True

def init_log_file():
    """Инициализация CSV (если нужно явно)"""
    try:
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
                writer.writeheader()
    except Exception as e:
        logger.error(f"Error initializing log file: {e}")

def log_pair_to_csv(pair_data):
    """Сохранение строки в CSV"""
    try:
        file_exists = os.path.exists(LOG_FILE)
        mode = 'a' if file_exists else 'w'
        
        with open(LOG_FILE, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(pair_data)
    except Exception as e:
        logger.error(f"Error logging to CSV: {e}")

async def handle_new_pair(event, chain, w3):
    try:
        if len(event['topics']) < 4:
            logger.warning(f"Event has only {len(event['topics'])} topics, skipping...")
            return

        token0_hex = event['topics'][1].hex()
        token0_address = Web3.to_checksum_address('0x' + token0_hex[-40:])

        token1_hex = event['topics'][2].hex()
        token1_address = Web3.to_checksum_address('0x' + token1_hex[-40:])

        pair_hex = event['topics'][3].hex()
        pair_address = Web3.to_checksum_address('0x' + pair_hex[-40:])

        logger.info(f"New pair on {chain.upper()}: {pair_address} (token0={token0_address}, token1={token1_address})")

        # Определяем, какой из токенов "новый"
        # Предположим, если token0 в STABLE_TOKENS, тогда "новый" = token1
        # Иначе считаем, что "новый" = token0
        if token0_address in STABLE_TOKENS:
            token_address_to_analyze = token1_address
        else:
            token_address_to_analyze = token0_address

        # Пытаемся вычитать символ
        try:
            token_contract = w3.eth.contract(address=token_address_to_analyze, abi=TOKEN_ABI)
            token_symbol = token_contract.functions.symbol().call()
        except:
            token_symbol = "UNKNOWN"
        
        logger.info(f"Token symbol: {token_symbol}")

        # Запускаем анализ
        metrics = await analyze_token_metrics(token_address_to_analyze, chain, w3)
        
        # Если хотим условие "total_score >= 70"
        if metrics['total_score'] >= 70:
            logger.info(f"Promising token found: {token_symbol} score={metrics['total_score']}")
            
            pair_data = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'chain': chain,
                'token_address': token_address_to_analyze,
                'pair_address': pair_address,
                'token_symbol': token_symbol,
                'liquidity': 'N/A',  # для реальной ликвидности нужно getReserves из пары
                'honeypot': metrics['is_honeypot'],
                'rugpull_risk': metrics['is_proxy'],
                'social_activity': metrics['social_score'],
                'total_score': metrics['total_score'],
                'liquidity_score': metrics['liquidity_score'],
                'contract_score': metrics['contract_score'],
                'social_score': metrics['social_score'],
                'volume_score': metrics['volume_score'],
                'holder_score': metrics['holder_score']
            }
            log_pair_to_csv(pair_data)

            # Отправка в Telegram
            message = (
                f"🚨 Новая перспективная пара!\n\n"
                f"Сеть: {chain.upper()}\n"
                f"Токен: {token_symbol}\n"
                f"Адрес: {token_address_to_analyze}\n"
                f"Пара: {pair_address}\n\n"
                f"Суммарный скор: {metrics['total_score']}/100\n"
                f"Ликвидность: {metrics['liquidity_score']}\n"
                f"Контракт: {metrics['contract_score']}\n"
                f"Социальные: {metrics['social_score']}\n"
                f"Объем: {metrics['volume_score']}\n"
                f"Холдеры: {metrics['holder_score']}\n"
            )
            
            await telegram_bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
            logger.info(f"Telegram notification sent for {token_symbol}")
            
    except Exception as e:
        logger.error(f"Error handling new pair: {e}", exc_info=True)

def get_social_signals(token_address):
    """Заглушка. Сюда можно поставить реальную логику: Twitter API, TG-каналы, etc."""
    return {"twitter_mentions": 0, "telegram_users": 0, "overall_sentiment": "neutral"}

async def get_events_in_chunks(
    w3, factory_address, from_block, to_block, pair_created_topic, chain
):
    chunk_size = 100
    current_block = from_block

    while current_block <= to_block:
        end_block = min(current_block + chunk_size, to_block)
        logger.info(
            f"Fetching {chain.upper()} events from block {current_block} to {end_block} (chunk_size={chunk_size})"
        )
        try:
            events = w3.eth.get_logs({
                'address': factory_address,
                'fromBlock': current_block,
                'toBlock': end_block,
                'topics': [pair_created_topic]
            })
            for event in events:
                await handle_new_pair(event, chain, w3)
            current_block = end_block + 1
            await asyncio.sleep(1)

        except Exception as e:
            if isinstance(e, ValueError) and e.args:
                err_data = e.args[0]
                if isinstance(err_data, dict) and err_data.get('code') == -32005:
                    logger.warning(f"Limit exceeded on chunk_size={chunk_size}, halving...")
                    if chunk_size > 1:
                        chunk_size = max(1, chunk_size // 2)
                    else:
                        logger.warning("Already at chunk_size=1; waiting 10s")
                        await asyncio.sleep(10)
                else:
                    logger.error(f"Unknown error: {e}", exc_info=True)
                    await asyncio.sleep(5)
            else:
                logger.error(f"Error fetching logs: {e}", exc_info=True)
                await asyncio.sleep(5)
            # Пробуем ещё раз с тем же current_block
            continue

async def analyze_token_metrics(token_address, chain, w3):
    """
    Расширенный анализ. Возвращаем словарь со всеми показателями.
    """
    try:
        metrics = {
            "is_honeypot": False,
            "is_proxy": False,
            "total_score": 0,
            "liquidity_score": 0,
            "contract_score": 0,
            "social_score": 0,
            "volume_score": 0,
            "holder_score": 0
        }
        
        metrics["is_honeypot"] = check_honeypot(token_address, chain)
        if metrics["is_honeypot"]:
            logger.warning(f"Honeypot detected for token {token_address}")
            return metrics
            
        metrics["is_proxy"] = check_rugpull(token_address, w3)
        if metrics["is_proxy"]:
            logger.warning(f"Potential rugpull for token {token_address}")
            return metrics

        # Соц. показатели
        social_data = get_social_signals(token_address)
        # Пример крайне упрощённой логики
        if social_data["twitter_mentions"] > 1000 or social_data["telegram_users"] > 5000:
            metrics["social_score"] = 20
        elif social_data["twitter_mentions"] > 500 or social_data["telegram_users"] > 2000:
            metrics["social_score"] = 10
        
        # Анализ контракта (наивный)
        try:
            contract_code = w3.eth.get_code(token_address).hex().lower()
            suspicious_patterns = ["selfdestruct", "delegatecall", "transfer.owner"]
            contract_risk = 0
            for pattern in suspicious_patterns:
                if pattern in contract_code:
                    contract_risk += 10
            # Допустим максимальный балл 30, вычитая risk
            base_score = 30
            contract_score = max(0, base_score - contract_risk)
            # Чтобы всё было в одном масштабе, приведём к 20 (например)
            metrics["contract_score"] = round(contract_score * (20 / 30))
        except Exception as e:
            logger.error(f"Error analyzing contract code: {e}")
            metrics["contract_score"] = 0
        
        # Оценка ликвидности
        # (здесь заглушка: делаем вид, что ликвидность=50000)
        liquidity = 50000
        metrics["liquidity_score"] = 20 if liquidity >= 50000 else 10
        
        # volume_score и holder_score (заглушки)
        metrics["volume_score"] = 10
        metrics["holder_score"] = 10

        metrics["total_score"] = sum([
            metrics["liquidity_score"],
            metrics["contract_score"],
            metrics["social_score"],
            metrics["volume_score"],
            metrics["holder_score"]
        ])

        return metrics
        
    except Exception as e:
        logger.error(f"Error in analyze_token_metrics: {e}", exc_info=True)
        return {
            "is_honeypot": True,
            "is_proxy": True,
            "total_score": 0,
            "liquidity_score": 0,
            "contract_score": 0,
            "social_score": 0,
            "volume_score": 0,
            "holder_score": 0
        }

async def main():
    logger.info("Starting DEX bot...")
    
    # Инициализируем Web3
    try:
        w3_eth = init_web3(INFURA_URL)
        w3_bsc = init_web3(BSC_NODE_URL, is_poa=True)
        logger.info("Web3 providers initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Web3: {e}", exc_info=True)
        return

    # Контракты
    try:
        uniswap_factory = w3_eth.eth.contract(
            address=Web3.to_checksum_address(UNISWAP_FACTORY_ADDRESS),
            abi=factory_abi
        )
        
        pancake_factory = w3_bsc.eth.contract(
            address=Web3.to_checksum_address(PANCAKE_FACTORY_ADDRESS),
            abi=factory_abi
        )
        logger.info("DEX factory contracts initialized")
    except Exception as e:
        logger.error(f"Failed to initialize DEX contracts: {e}", exc_info=True)
        return

    pair_created_topic = Web3.keccak(text="PairCreated(address,address,address,uint256)")

    # Начальные блоки
    eth_latest_block = w3_eth.eth.block_number - 100
    bsc_latest_block = w3_bsc.eth.block_number - 100

    if eth_latest_block < 0: eth_latest_block = 0
    if bsc_latest_block < 0: bsc_latest_block = 0

    logger.info(f"Starting from blocks: ETH={eth_latest_block}, BSC={bsc_latest_block}")

    while True:
        try:
            current_eth_block = w3_eth.eth.block_number
            if current_eth_block > eth_latest_block:
                await get_events_in_chunks(
                    w3_eth,
                    uniswap_factory.address,
                    eth_latest_block + 1,
                    current_eth_block,
                    pair_created_topic,
                    'eth'
                )
                eth_latest_block = current_eth_block

            current_bsc_block = w3_bsc.eth.block_number
            if current_bsc_block > bsc_latest_block:
                await get_events_in_chunks(
                    w3_bsc,
                    pancake_factory.address,
                    bsc_latest_block + 1,
                    current_bsc_block,
                    pair_created_topic,
                    'bsc'
                )
                bsc_latest_block = current_bsc_block

            await asyncio.sleep(60)

        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
            # Переподключение
            w3_eth = init_web3(INFURA_URL)
            w3_bsc = init_web3(BSC_NODE_URL, is_poa=True)
            await asyncio.sleep(60)

if __name__ == "__main__":
    init_log_file()  # если нужно
    logger.info("Bot starting...")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot stopped due to error: {e}", exc_info=True)