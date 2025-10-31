import requests
import asyncio
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware  # Исправленный импорт
from web3.types import BlockParams
from web3.providers.persistent import WebSocketProvider
import time
import json
from telegram import Bot
import csv
import os
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler

# Настройка логирования
def setup_logging():
    """Настройка системы логирования"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, "DEX_bot.log")
    
    # Создаем форматтер для логов
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Настраиваем файловый handler с ротацией
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    
    # Настраиваем консольный handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Настраиваем корневой логгер
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Получаем хеш события PairCreated
pair_created_topic = Web3.keccak(text="PairCreated(address,address,address,uint256)")

async def listen_for_pairs():
    # Подключаемся к WebSocket (пример ниже — бесплатные узлы, но лучше свой/платный)
    w3_ws = Web3(WebSocketProvider('wss://bsc-ws-node.nariox.org:443'))

    # Создаём фильтр
    event_filter = w3_ws.eth.filter({
        'address': PANCAKE_FACTORY_ADDRESS,
        'topics': [pair_created_topic]
    })

    while True:
        try:
            for event in event_filter.get_new_entries():
                await handle_new_pair(event, 'bsc')
            await asyncio.sleep(2)
        except Exception as e:
            logger.error(f"WebSocket error: {e}", exc_info=True)
            # Переподключаемся при ошибке
            await asyncio.sleep(10)
            w3_ws = Web3(WebSocketProvider('wss://bsc-ws-node.nariox.org:443'))
            event_filter = w3_ws.eth.filter({
                'address': PANCAKE_FACTORY_ADDRESS,
                'topics': [pair_created_topic]
            })

# Инициализация логгера
logger = setup_logging()

# Конфигурация
TELEGRAM_TOKEN = '7638758608:AAF3awK3NRisz5dzCxfK2jMVC26W6D2DV-E'
TELEGRAM_CHAT_ID = '36972091'
INFURA_URL = 'https://mainnet.infura.io/v3/0be54e0961d04eb486cef5731b191b3f'
BSC_NODE_URL = 'https://bsc-dataseed.binance.org/'
UNISWAP_FACTORY_ADDRESS = '0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f'
PANCAKE_FACTORY_ADDRESS = '0xcA143Ce32Fe78f1f7019d7d551a6402fC5350c73'
LOG_FILE = "new_pairs_log.csv"

# Добавляем константы для логирования
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

# Инициализация Telegram бота
telegram_bot = Bot(token=TELEGRAM_TOKEN)

# Инициализация Web3
w3_eth = Web3(Web3.HTTPProvider(INFURA_URL))
w3_bsc = Web3(Web3.HTTPProvider(BSC_NODE_URL))
w3_bsc.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)

# ABI для фабрики Uniswap/Pancake
factory_abi = json.loads('[{"constant":true,"inputs":[],"name":"getPair","outputs":[{"internalType":"address","name":"","type":"address"}],"payable":false,"stateMutability":"view","type":"function"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"token0","type":"address"},{"indexed":true,"internalType":"address","name":"token1","type":"address"},{"indexed":false,"internalType":"address","name":"pair","type":"address"},{"indexed":false,"internalType":"uint256","name":"","type":"uint256"}],"name":"PairCreated","type":"event"}]')

# Добавляем необходимые ABI
PAIR_ABI = [
    {
        "constant": True,
        "inputs": [],
        "name": "getReserves",
        "outputs": [
            {"name": "_reserve0", "type": "uint112"},
            {"name": "_reserve1", "type": "uint112"},
            {"name": "_blockTimestampLast", "type": "uint32"}
        ],
        "payable": False,
        "stateMutability": "view",
        "type": "function"
    }
]

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

def init_web3(provider_url, is_poa=False):
    w3 = Web3(Web3.HTTPProvider(provider_url))
    if is_poa:
        w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
    return w3

w3_eth = init_web3(INFURA_URL)
w3_bsc = init_web3(BSC_NODE_URL, is_poa=True)

def check_honeypot(token_address, chain):
    """Проверка на Honeypot через API"""
    try:
        if chain == 'eth':
            url = f'https://api.honeypot.is/v2/IsHoneypot?address={token_address}'
        else:
            url = f'https://api.honeypot.is/v2/IsHoneypot?address={token_address}&chainId=56'
        
        response = requests.get(url)
        data = response.json()
        if 'honeypotResult' in data:
            return data['honeypotResult']['isHoneypot']
        return False
    except Exception as e:
        print(f"Honeypot check error: {e}")
        return False

def check_rugpull(token_address):
    """Базовая проверка на Rugpull (пример)"""
    try:
        contract_code = w3_eth.eth.get_code(token_address)
        return len(contract_code) <= 2  # Упрощенная проверка
    except:
        return False

def check_social_links(token_symbol):
    """Проверка наличия соцсетей (пример)"""
    try:
        twitter_url = f"https://twitter.com/search?q={token_symbol}"
        response = requests.get(twitter_url)
        return "не найдено" not in response.text
    except:
        return False

def analyze_token(token_address, chain):
    """Основная функция проверки токена"""
    honeypot = check_honeypot(token_address, chain)
    rugpull_risk = check_rugpull(token_address)
    social_activity = check_social_links("EXAMPLE")  # Заменить на реальный символ
    
    return {
        'honeypot': honeypot,
        'rugpull_risk': rugpull_risk,
        'social_activity': social_activity
    }

def init_log_file():
    """Инициализация файла логов"""
    try:
        with open(LOG_FILE, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            writer.writeheader()
    except Exception as e:
        print(f"Error initializing log file: {e}")

def log_pair_to_csv(pair_data):
    """Сохранение данных о паре в CSV"""
    try:
        file_exists = os.path.exists(LOG_FILE)
        mode = 'a' if file_exists else 'w'
        
        with open(LOG_FILE, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(pair_data)
    except Exception as e:
        print(f"Error logging to CSV: {e}")

async def handle_new_pair(event, chain):
    """Обработка нового события создания пары"""
    try:
        # Получаем адрес токена из topics[1] и преобразуем его в правильный формат
        token_address = Web3.to_checksum_address('0x' + event['topics'][1].hex())
        # Получаем адрес пары из data и преобразуем его в правильный формат
        pair_address = Web3.to_checksum_address('0x' + event['data'][:66])
        
        logger.info(f"New pair detected on {chain.upper()}: {token_address}")
        
        # Получаем символ токена
        token_contract = w3_bsc.eth.contract(address=token_address, abi=TOKEN_ABI)
        token_symbol = token_contract.functions.symbol().call()
        
        logger.info(f"Token symbol: {token_symbol}")
        
        # Анализируем токен
        metrics = await analyze_token_metrics(token_address, chain)
        
        # Проверяем, является ли токен перспективным
        if metrics['total_score'] >= 70:
            logger.info(f"Promising token found: {token_symbol} with score {metrics['total_score']}")
            
            # Формируем данные для логирования
            pair_data = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'chain': chain,
                'token_address': token_address,
                'pair_address': pair_address,
                'token_symbol': token_symbol,
                'is_honeypot': metrics['is_honeypot'],
                'is_proxy': metrics['is_proxy'],
                'owner_renounced': metrics['owner_renounced'],
                'total_score': metrics['total_score'],
                'liquidity_score': metrics['liquidity_score'],
                'contract_score': metrics['contract_score'],
                'social_score': metrics['social_score'],
                'volume_score': metrics['volume_score'],
                'holder_score': metrics['holder_score']
            }
            
            # Сохраняем в CSV
            log_pair_to_csv(pair_data)
            logger.info(f"Pair data saved to CSV: {token_symbol}")
            
            # Отправляем уведомление
            message = f"🚨 Новая перспективная пара!\n\n"
            message += f"Сеть: {chain.upper()}\n"
            message += f"Токен: {token_symbol}\n"
            message += f"Адрес: {token_address}\n"
            message += f"Пара: {pair_address}\n\n"
            message += f"Оценка: {metrics['total_score']}/100\n"
            message += f"Ликвидность: {metrics['liquidity_score']}/20\n"
            message += f"Контракт: {metrics['contract_score']}/20\n"
            message += f"Социальные: {metrics['social_score']}/20\n"
            message += f"Объем: {metrics['volume_score']}/20\n"
            message += f"Холдеры: {metrics['holder_score']}/20\n"
            
            await telegram_bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
            logger.info(f"Telegram notification sent for {token_symbol}")
            
    except Exception as e:
        logger.error(f"Error handling new pair: {e}", exc_info=True)

def get_social_signals(token_address):
    """Получает социальные метрики через LunarCrush API"""
    LUNARCRUSH_API_KEY = "q5i7ctfjkdblg946f2p09wlq19m8bqddibkpf88b7"
    endpoint = "https://api.lunarcrush.com/v2"
    
    try:
        # Получаем символ токена
        token_contract = w3_bsc.eth.contract(
            address=Web3.to_checksum_address(token_address),
            abi=TOKEN_ABI
        )
        token_symbol = token_contract.functions.symbol().call()
        
        params = {
            "data": "assets",
            "key": LUNARCRUSH_API_KEY,
            "symbol": token_symbol
        }
        
        response = requests.get(endpoint, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get("data"):
                asset_data = data["data"][0]
                return {
                    "twitter_mentions": asset_data.get("twitter_mentions", 0),
                    "telegram_users": asset_data.get("telegram_members", 0),
                    "overall_sentiment": asset_data.get("social_score", "neutral")
                }
        return {"twitter_mentions": 0, "telegram_users": 0, "overall_sentiment": "neutral"}
    except Exception as e:
        print(f"Error getting social signals: {e}")
        return {"twitter_mentions": 0, "telegram_users": 0, "overall_sentiment": "neutral"}

async def get_events_in_chunks(
    w3, 
    factory_address, 
    from_block, 
    to_block, 
    pair_created_topic, 
    chain
):
    """Получение событий небольшими чанками с динамическим уменьшением размерности при ошибках."""

    # Начинаем с chunk_size в 500 (или 200 — подобрать эмпирически)
    chunk_size = 500  
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
            # Если запрос успешен — обрабатываем события
            for event in events:
                await handle_new_pair(event, chain)
            # Переходим к следующему чанку
            current_block = end_block + 1

        except Exception as e:
            # Перехватываем ошибку 'limit exceeded'
            if isinstance(e, ValueError) and e.args and len(e.args) > 0:
                error_data = e.args[0]
                # Проверяем, есть ли в сообщении 'limit exceeded' или код -32005
                if isinstance(error_data, dict) and error_data.get('code') == -32005:
                    logger.warning(
                        f"Limit exceeded on chunk_size={chunk_size}, "
                        f"reducing chunk size and retrying..."
                    )
                    # Уменьшаем размер чанка вдвое (хотя бы до 1)
                    if chunk_size > 1:
                        chunk_size = max(1, chunk_size // 2)
                    else:
                        # Если уже 1 блок — значит, просто даём узлу остыть
                        await asyncio.sleep(10)
                else:
                    # Если какая-то другая ошибка, логируем и ждём
                    logger.error(f"Unknown error fetching {chain.upper()} events: {e}", exc_info=True)
                    await asyncio.sleep(5)
            else:
                # Ошибка, не связанная с -32005
                logger.error(f"Error fetching {chain.upper()} events: {e}", exc_info=True)
                await asyncio.sleep(5)


async def main():
    logger.info("Starting DEX bot...")
    
    # Инициализация Web3 провайдеров
    try:
        w3_eth = init_web3(INFURA_URL)
        w3_bsc = init_web3(BSC_NODE_URL, is_poa=True)
        logger.info("Web3 providers initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Web3 providers: {e}", exc_info=True)
        return

    # Инициализация контрактов
    try:
        uniswap_factory = w3_eth.eth.contract(
            address=Web3.to_checksum_address(UNISWAP_FACTORY_ADDRESS),
            abi=factory_abi
        )
        
        pancake_factory = w3_bsc.eth.contract(
            address=Web3.to_checksum_address(PANCAKE_FACTORY_ADDRESS),
            abi=factory_abi
        )
        logger.info("DEX factory contracts initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize DEX contracts: {e}", exc_info=True)
        return

    # Получаем текущий блок
    eth_latest_block = w3_eth.eth.block_number
    bsc_latest_block = w3_bsc.eth.block_number
    logger.info(f"Starting from blocks - ETH: {eth_latest_block}, BSC: {bsc_latest_block}")

    while True:
        try:
            # Для Ethereum
            eth_current_block = w3_eth.eth.block_number
            if eth_current_block > eth_latest_block:
                await get_events_in_chunks(
                    w3_eth,
                    uniswap_factory.address,
                    eth_latest_block + 1,
                    eth_current_block,
                    pair_created_topic,
                    'eth'
                )
                eth_latest_block = eth_current_block

            # Для BSC
            bsc_current_block = w3_bsc.eth.block_number
            if bsc_current_block > bsc_latest_block:
                await get_events_in_chunks(
                    w3_bsc,
                    pancake_factory.address,
                    bsc_latest_block + 1,
                    bsc_current_block,
                    pair_created_topic,
                    'bsc'
                )
                bsc_latest_block = bsc_current_block

            await asyncio.sleep(60)  # Задержка между основными циклами
        
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
            # Переподключение обоих провайдеров
            w3_eth = init_web3(INFURA_URL)
            w3_bsc = init_web3(BSC_NODE_URL, is_poa=True)
            await asyncio.sleep(60)  # Задержка перед повторной попыткой

async def analyze_token_metrics(token_address, chain):
    """Расширенный анализ метрик токена"""
    try:
        # Инициализация Web3 в зависимости от сети
        w3 = w3_eth if chain == 'eth' else w3_bsc
        
        # Базовые метрики
        metrics = {
            "is_honeypot": False,
            "is_proxy": False,
            "owner_renounced": False,
            "total_score": 0,
            "liquidity_score": 0,
            "contract_score": 0,
            "social_score": 0,
            "volume_score": 0,
            "holder_score": 0
        }
        
        # Проверка на honeypot
        metrics["is_honeypot"] = check_honeypot(token_address, chain)
        if metrics["is_honeypot"]:
            logger.warning(f"Honeypot detected for token {token_address}")
            return metrics
            
        # Проверка на rugpull
        metrics["is_proxy"] = check_rugpull(token_address)
        if metrics["is_proxy"]:
            logger.warning(f"Potential rugpull detected for token {token_address}")
            return metrics
            
        # Проверка социальной активности
        social_data = get_social_signals(token_address)
        if social_data["twitter_mentions"] > 1000 or social_data["telegram_users"] > 5000:
            metrics["social_score"] = 20
        elif social_data["twitter_mentions"] > 500 or social_data["telegram_users"] > 2000:
            metrics["social_score"] = 10
            
        # Анализ контракта
        try:
            contract_code = w3.eth.get_code(token_address).hex()
            suspicious_patterns = [
                "selfdestruct",
                "delegatecall",
                "transfer.owner"
            ]
            
            contract_risk = 0
            for pattern in suspicious_patterns:
                if pattern in contract_code.lower():
                    contract_risk += 10
                    
            metrics["contract_score"] = max(0, 30 - contract_risk)
            
        except Exception as e:
            logger.error(f"Error analyzing contract: {e}")
            metrics["contract_score"] = 0
            
        # Оценка ликвидности
        try:
            pair_contract = w3.eth.contract(address=token_address, abi=PAIR_ABI)
            reserves = pair_contract.functions.getReserves().call()
            liquidity = sum(reserves)
            
            if liquidity >= 50000:
                metrics["liquidity_score"] = 30
            elif liquidity >= 20000:
                metrics["liquidity_score"] = 20
            elif liquidity >= 5000:
                metrics["liquidity_score"] = 10
                
        except Exception as e:
            logger.error(f"Error checking liquidity: {e}")
            metrics["liquidity_score"] = 0
            
        # Оценка объема
        try:
            if liquidity > 50000:
                metrics["volume_score"] = 10
            elif liquidity > 20000:
                metrics["volume_score"] = 5
                
        except Exception as e:
            logger.error(f"Error checking volume: {e}")
            metrics["volume_score"] = 0
            
        # Оценка холдеров
        try:
            if social_data["telegram_users"] > 1000:
                metrics["holder_score"] = 10
            elif social_data["telegram_users"] > 500:
                metrics["holder_score"] = 5
                
        except Exception as e:
            logger.error(f"Error checking holders: {e}")
            metrics["holder_score"] = 0
            
        # Суммарная оценка
        metrics["total_score"] = sum([
            metrics["liquidity_score"],
            metrics["contract_score"],
            metrics["social_score"],
            metrics["volume_score"],
            metrics["holder_score"]
        ])
        
        logger.info(f"Token {token_address} analysis completed. Total score: {metrics['total_score']}")
        return metrics
        
    except Exception as e:
        logger.error(f"Error in analyze_token_metrics: {e}", exc_info=True)
        return {
            "is_honeypot": True,
            "is_proxy": True,
            "owner_renounced": False,
            "total_score": 0,
            "liquidity_score": 0,
            "contract_score": 0,
            "social_score": 0,
            "volume_score": 0,
            "holder_score": 0
        }

if __name__ == "__main__":
    logger.info("Bot starting...")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot stopped due to error: {e}", exc_info=True)
