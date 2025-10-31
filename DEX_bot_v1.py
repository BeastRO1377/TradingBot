import os
import time
import csv
import json
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime
from web3 import Web3
from web3.exceptions import ContractLogicError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import random
from functools import lru_cache

# region Конфигурация
# ==============================================================================
time.sleep(random.uniform(0.5, 1.5))  # Между 0.5 и 1.5 секунд

@lru_cache(maxsize=100)
def get_token_metadata_cached(token_address):
    return get_token_metadata(token_address)


BSC_RPC_URLS = [
    "https://bsc-dataseed.binance.org/",
    "https://bsc-dataseed1.binance.org/",
    "https://bsc-dataseed2.binance.org/",
    "https://bsc-dataseed3.binance.org/",
    "https://bsc-dataseed4.binance.org/",
    "https://bsc-dataseed1.defibit.io/",
    "https://bsc-dataseed2.defibit.io/",
    "https://bsc-dataseed3.defibit.io/",
    "https://bsc-dataseed4.defibit.io/",
    "https://bsc-dataseed1.ninicoin.io/",
    "https://bsc-dataseed2.ninicoin.io/",
    "https://bsc-dataseed3.ninicoin.io/",
    "https://bsc-dataseed4.ninicoin.io/"
]

LUNARCRUSH_API_KEY = "q5i7ctfjkdblg946f2p09wlq19m8bqddibkpf88b7"
MIN_LIQUIDITY_USD = 5000
LOG_FILE = "new_pairs_log.csv"
LOG_ROTATION_SIZE = 104857600
BACKUP_DIR = "log_backups"

MAX_RETRIES = 5  # Максимальное количество попыток для одного чанка
INITIAL_CHUNK_SIZE = 100  # Начальный размер чанка
MIN_CHUNK_SIZE = 10  # Минимальный размер чанка
BASE_DELAY = 2  # Базовая задержка между попытками


PANCAKE_FACTORY_ADDRESS = "0xBCfCcbde45cE874adCB698cC183deBcF17952812"
WBNB_ADDRESS = "0xbb4CdB9CBd36B01BD1cBaEBF2De08d9173bc095c"
USDT_ADDRESS = "0x55d398326f99059fF775485246999027B3197955"

FACTORY_ABI = [{
    "anonymous": False,
    "inputs": [
        {"indexed": True, "name": "token0", "type": "address"},
        {"indexed": True, "name": "token1", "type": "address"},
        {"indexed": False, "name": "pair", "type": "address"}
    ],
    "name": "PairCreated",
    "type": "event"
}]

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
    },
    {
        "constant": True,
        "inputs": [],
        "name": "token0",
        "outputs": [{"name": "", "type": "address"}],
        "payable": False,
        "stateMutability": "view",
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "token1",
        "outputs": [{"name": "", "type": "address"}],
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
    },
    {
        "constant": True,
        "inputs": [],
        "name": "name",
        "outputs": [{"name": "", "type": "string"}],
        "payable": False,
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "payable": False,
        "type": "function"
    }
]

LOG_FIELDS = [
    'discovery_time',
    'pair_address',
    'token0',
    'symbol0',
    'token1',
    'symbol1',
    'liquidity_estimated_usd',
    'safety_score',
    'rugdoc_status',
    'social_twitter',
    'social_telegram',
    'sentiment'
]
# ==============================================================================
# endregion

# region Инициализация
# ==============================================================================
class Web3Provider:
    def __init__(self, rpc_urls):
        self.rpc_urls = rpc_urls
        self.current_index = 0
        self.web3 = self._connect()

    def _connect(self):
        while self.current_index < len(self.rpc_urls):
            try:
                provider = Web3.HTTPProvider(self.rpc_urls[self.current_index])
                web3 = Web3(provider)
                if web3.is_connected():
                    print(f"Connected to {self.rpc_urls[self.current_index]}")
                    return web3
                self.current_index += 1
            except:
                self.current_index += 1
        raise ConnectionError("All RPC nodes are unavailable")

    def get_instance(self):
        if not self.web3.is_connected():
            self.web3 = self._connect()
        return self.web3
    
    def reconnect(self):
        self.current_index = (self.current_index + 1) % len(self.rpc_urls)
        self.web3 = self._connect()
        return self.web3

web3_provider = Web3Provider(BSC_RPC_URLS)

session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
# ==============================================================================
# endregion

# region Вспомогательные функции
# ==============================================================================
def get_bnb_price():
    """Получает текущую цену BNB через CoinGecko API"""
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {"ids": "binancecoin", "vs_currencies": "usd"}
    
    try:
        response = session.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data["binancecoin"]["usd"]
    except Exception as e:
        print(f"Error getting BNB price: {e}, using fallback value")
        return 300

def init_log_system():
    """Инициализирует систему логирования"""
    Path(BACKUP_DIR).mkdir(exist_ok=True)
    
    if not Path(LOG_FILE).exists():
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            writer.writeheader()

def rotate_logs():
    """Ротация лог-файлов"""
    if Path(LOG_FILE).stat().st_size > LOG_ROTATION_SIZE:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = Path(BACKUP_DIR) / f"pairs_{timestamp}.csv"
        Path(LOG_FILE).rename(backup_file)
        init_log_system()

def log_to_csv(dataframe):
    """Логирует DataFrame в CSV"""
    try:
        rotate_logs()
        dataframe["discovery_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = not Path(LOG_FILE).exists()
        
        dataframe[LOG_FIELDS].to_csv(
            LOG_FILE,
            mode="a",
            header=header,
            index=False,
            encoding="utf-8"
        )
        print(f"Logged {len(dataframe)} pairs to {LOG_FILE}")
    except Exception as e:
        print(f"Logging failed: {e}")

def get_token_metadata(token_address):
    """Получает метаданные токена"""
    web3 = web3_provider.get_instance()
    contract = web3.eth.contract(
        address=web3.to_checksum_address(token_address),
        abi=TOKEN_ABI
    )
    
    metadata = {
        "symbol": "UNKNOWN",
        "name": "Unknown Token",
        "decimals": 18,
        "address": token_address
    }
    
    try:
        metadata["symbol"] = contract.functions.symbol().call()
    except (ContractLogicError, OverflowError, ValueError):
        pass
    
    try:
        metadata["name"] = contract.functions.name().call()
    except (ContractLogicError, OverflowError, ValueError):
        pass
    
    try:
        metadata["decimals"] = contract.functions.decimals().call()
    except (ContractLogicError, OverflowError, ValueError):
        pass
    
    return metadata

def get_pair_reserves(pair_address):
    """Получение резервов с повторными попытками"""
    web3 = web3_provider.get_instance()
    for _ in range(3):
        try:
            pair_contract = web3.eth.contract(address=pair_address, abi=PAIR_ABI)
            reserves = pair_contract.functions.getReserves().call()
            return reserves[0], reserves[1]
        except:
            time.sleep(1)
    return 0, 0

def check_honeypot_or_scam(token_address):
    """Проверка токена через RugDoc API"""
    chain = "&chain=bsc"
    honeypot_url = "https://honeypot.api.rugdoc.io/api/honeypotStatus.js?address="
    url = honeypot_url + token_address + chain

    try:
        resp = session.get(url, timeout=10)
        if resp.status_code != 200:
            return 10, False, f"API error: {resp.status_code}"
            
        data = resp.json()
        status = data.get("status", "UNKNOWN")
        interpretations = {
            "UNKNOWN": (False, "Unknown status"),
            "OK": (True, "Token is safe"),
            "NO_PAIRS": (False, "No pairs found"),
            "SEVERE_FEE": (False, "High fee >50%"),
            "HIGH_FEE": (True, "High fee 20-50%"),
            "MEDIUM_FEE": (True, "Medium fee 10-20%"),
            "APPROVE_FAILED": (False, "Approve failed"),
            "SWAP_FAILED": (False, "Swap failed")
        }
        is_safe, desc = interpretations.get(status, (False, "Unrecognized status"))
        return (70 if is_safe else 10), is_safe, desc
    except Exception as e:
        return 10, False, f"Check failed: {str(e)}"

def get_social_signals(token_symbol):
    """Получает социальные метрики через LunarCrush"""
    endpoint = "https://api.lunarcrush.com/v2"
    params = {
        "data": "assets",
        "key": LUNARCRUSH_API_KEY,
        "symbol": token_symbol
    }
    
    try:
        response = session.get(endpoint, params=params, timeout=10)
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
    except:
        return {"twitter_mentions": 0, "telegram_users": 0, "overall_sentiment": "neutral"}

def calculate_liquidity(reserves, token0_meta, token1_meta, bnb_price):
    """Рассчитывает ликвидность в USD"""
    web3 = web3_provider.get_instance()
    wbnb = web3.to_checksum_address(WBNB_ADDRESS)
    usdt = web3.to_checksum_address(USDT_ADDRESS)
    
    if token0_meta["address"] == wbnb:
        return (reserves[0] / 10**token0_meta["decimals"]) * bnb_price * 2
    elif token1_meta["address"] == wbnb:
        return (reserves[1] / 10**token1_meta["decimals"]) * bnb_price * 2
    elif token0_meta["address"] == usdt:
        return (reserves[0] / 10**token0_meta["decimals"]) * 2
    elif token1_meta["address"] == usdt:
        return (reserves[1] / 10**token1_meta["decimals"]) * 2
    return 0

def analyze_token_metrics(token_address, liquidity, web3):
    """Расширенный анализ метрик токена"""
    score = 0
    metrics = {
        "liquidity_score": 0,
        "contract_score": 0,
        "social_score": 0,
        "volume_score": 0,
        "holder_score": 0
    }
    
    # Оценка ликвидности (0-30 баллов)
    if liquidity >= 50000:
        metrics["liquidity_score"] = 30
    elif liquidity >= 20000:
        metrics["liquidity_score"] = 20
    elif liquidity >= 5000:
        metrics["liquidity_score"] = 10
        
    # Анализ контракта (0-30 баллов)
    try:
        contract_code = web3.eth.get_code(token_address).hex()
        
        # Проверка на наличие подозрительных функций
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
        
    except Exception:
        metrics["contract_score"] = 0
        
    # Социальная активность (0-20 баллов)
    social_data = get_social_signals(token_address)
    if social_data["twitter_mentions"] > 1000 or social_data["telegram_users"] > 5000:
        metrics["social_score"] = 20
    elif social_data["twitter_mentions"] > 500 or social_data["telegram_users"] > 2000:
        metrics["social_score"] = 10
        
    # Объем торгов (0-10 баллов)
    try:
        pair_contract = web3.eth.contract(address=token_address, abi=PAIR_ABI)
        events = pair_contract.events.Swap.get_logs(
            fromBlock=web3.eth.block_number - 1000
        )
        daily_volume = sum(event["args"]["amount0In"] + event["args"]["amount1In"] 
                         for event in events)
        
        if daily_volume > 50000:
            metrics["volume_score"] = 10
        elif daily_volume > 20000:
            metrics["volume_score"] = 5
            
    except Exception:
        metrics["volume_score"] = 0
        
    # Распределение холдеров (0-10 баллов)
    try:
        token_contract = web3.eth.contract(address=token_address, abi=TOKEN_ABI)
        transfer_events = token_contract.events.Transfer.get_logs(
            fromBlock=web3.eth.block_number - 10000
        )
        
        unique_holders = len(set(event["args"]["to"] for event in transfer_events))
        if unique_holders > 100:
            metrics["holder_score"] = 10
        elif unique_holders > 50:
            metrics["holder_score"] = 5
            
    except Exception:
        metrics["holder_score"] = 0
        
    # Суммарная оценка
    score = sum(metrics.values())
    
    return score, metrics

def alert_on_promising_pair(pair_data):
    """Оповещение о найденной перспективной паре"""
    if pair_data["total_score"] >= 80:
        message = f"""
🚀 Найдена перспективная торговая пара!

Пара: {pair_data['symbol0']}/{pair_data['symbol1']}
Адрес: {pair_data['pair_address']}
Ликвидность: ${pair_data['liquidity_estimated_usd']:,.2f}
Общий скор: {pair_data['total_score']}/100

Метрики:
- Ликвидность: {pair_data['liquidity_score']}/30
- Безопасность контракта: {pair_data['contract_score']}/30
- Социальная активность: {pair_data['social_score']}/20
- Объем торгов: {pair_data['volume_score']}/10
- Распределение холдеров: {pair_data['holder_score']}/10

Рекомендация: {pair_data['recommendation']}
        """
        print(message)
        # Здесь можно добавить отправку в Telegram или другие каналы
# ==============================================================================
# endregion

# region Основная логика
# ==============================================================================
def analyze_new_pairs(from_block, to_block):
    """Анализирует новые пары в блокчейне с адаптивными лимитами"""
    logs = []
    current_from = from_block
    chunk_size = INITIAL_CHUNK_SIZE
    attempt = 0
    
    while current_from <= to_block:
        current_to = min(current_from + chunk_size - 1, to_block)
        web3 = web3_provider.get_instance()
        factory_contract = web3.eth.contract(
            address=web3.to_checksum_address(PANCAKE_FACTORY_ADDRESS),
            abi=FACTORY_ABI
        )

        try:
            # Увеличиваем задержку при повторных попытках
            time.sleep(BASE_DELAY * (2 ** attempt))
            
            chunk_logs = web3.eth.get_logs({
                "fromBlock": current_from,
                "toBlock": current_to,
                "address": factory_contract.address,
                "topics": [web3.keccak(text="PairCreated(address,address,address,uint256)")]
            })
            
            logs.extend(chunk_logs)
            current_from += chunk_size
            chunk_size = INITIAL_CHUNK_SIZE  # Сброс размера чанка после успеха
            attempt = 0  # Сброс счетчика попыток

        except Exception as e:
            error_msg = str(e)
            
            # Обработка ошибки превышения лимита
            if 'limit exceeded' in error_msg or '-32005' in error_msg:
                print(f"Rate limit hit, adjusting chunk size...")
                chunk_size = max(chunk_size // 2, MIN_CHUNK_SIZE)
                attempt += 1
                
                if attempt >= MAX_RETRIES:
                    print("Max retries reached, rotating RPC node...")
                    web3_provider.reconnect()
                    attempt = 0
                    time.sleep(10)
                
                continue
                
            # Обработка других ошибок
            print(f"Error fetching blocks {current_from}-{current_to}: {e}")
            time.sleep(10)
            web3_provider.reconnect()

    results = []
    bnb_price = get_bnb_price()
    
    for log in logs:
        event = factory_contract.events.PairCreated().processLog(log)
        token0 = web3.to_checksum_address(event.args.token0)
        token1 = web3.to_checksum_address(event.args.token1)
        pair = web3.to_checksum_address(event.args.pair)
        
        # Получение метаданных
        token0_meta = get_token_metadata(token0)
        token1_meta = get_token_metadata(token1)
        
        # Получение резервов
        reserves = get_pair_reserves(pair)
        
        # Расчет ликвидности
        liquidity = calculate_liquidity(reserves, token0_meta, token1_meta, bnb_price)
        
        # Проверка безопасности
        safety_score, is_safe, rugdoc_status = check_honeypot_or_scam(token0)
        
        # Социальные метрики
        social_data = get_social_signals(token0_meta["symbol"])
        
        # Расширенный анализ
        token_score, metrics = analyze_token_metrics(token0, liquidity, web3)
        
        # Дополнительные проверки
        is_valid_pair = (
            liquidity >= MIN_LIQUIDITY_USD and
            token_score >= 50 and  # Минимальный проходной балл
            metrics["contract_score"] >= 20  # Минимальная безопасность контракта
        )
        
        if is_valid_pair:
            results.append({
                "pair_address": pair,
                "token0": token0,
                "symbol0": token0_meta["symbol"],
                "token1": token1,
                "symbol1": token1_meta["symbol"],
                "liquidity_estimated_usd": round(liquidity, 2),
                "safety_score": safety_score,
                "rugdoc_status": rugdoc_status,
                "social_twitter": social_data["twitter_mentions"],
                "social_telegram": social_data["telegram_users"],
                "sentiment": social_data["overall_sentiment"],
                "total_score": token_score,
                "liquidity_score": metrics["liquidity_score"],
                "contract_score": metrics["contract_score"],
                "social_score": metrics["social_score"],
                "volume_score": metrics["volume_score"],
                "holder_score": metrics["holder_score"],
                "recommendation": "Strong Buy" if token_score >= 80 else "Consider Buy"
            })
    
    return pd.DataFrame(results)

def main_loop():
    """Улучшенный главный цикл"""
    init_log_system()
    
    while True:
        try:
            web3 = web3_provider.get_instance()
            last_block = web3.eth.block_number
            
            while True:
                current_block = web3.eth.block_number
                if current_block > last_block:
                    print(f"\n[{datetime.now()}] Анализ блоков {last_block+1}-{current_block}")
                    
                    pairs_df = analyze_new_pairs(last_block + 1, current_block)
                    last_block = current_block
                    
                    if not pairs_df.empty:
                        # Фильтрация по расширенным критериям
                        promising_pairs = pairs_df[
                            (pairs_df["total_score"] >= 50) &
                            (pairs_df["contract_score"] >= 20)
                        ]
                        
                        if not promising_pairs.empty:
                            print("\nНайдены перспективные пары:")
                            print(promising_pairs[
                                ['symbol0', 'symbol1', 'liquidity_estimated_usd',
                                 'total_score', 'recommendation']
                            ].to_string(index=False))
                            
                            # Оповещения о самых перспективных парах
                            for _, pair in promising_pairs.iterrows():
                                alert_on_promising_pair(pair)
                            
                            log_to_csv(promising_pairs)
                
                time.sleep(60)
        
        except Exception as e:
            print(f"Критическая ошибка: {e}")
            web3_provider.reconnect()
            time.sleep(30)
# ==============================================================================
# endregion

if __name__ == "__main__":
    main_loop()