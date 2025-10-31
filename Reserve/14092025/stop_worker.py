#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Автономный воркер трейлинг-стопа (АРХИТЕКТУРА V2 - БЕЗ WS).
- Получает последнюю цену из stdin.
- Отправляет события в stdout.
"""
import sys, json, math, time, asyncio

# --- УБИРАЕМ ИМПОРТ PYBIT ---

def jprint(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()

def log_err(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(f"{ts} [stop_worker] {msg}", file=sys.stderr, flush=True)

def f(x, d=0.0):
    try: return float(x)
    except: return d

class TrailingStopWorker:
    def __init__(self, p: dict):
        # ... (эта часть без изменений) ...
        self.symbol      = p.get("symbol")
        self.side        = (p.get("side") or "").capitalize()
        self.avg_price   = f(p.get("avg_price"))
        self.leverage    = f(p.get("leverage"), 10.0)
        self.tick_size   = f(p.get("tick_size"), 1e-6)
        self.start_roi   = f(p.get("start_roi"), 5.0)
        self.gap_mode    = p.get("gap_mode", "roi")
        self.gap_roi_pct = f(p.get("gap_roi_pct"), 1.0)
        self.hb_interval = f(p.get("hb_interval"), 15.0)
        self.initial_price = float(p.get("initial_price") or 0.0)
        self.trailing_activated = False
        self.current_stop = None
        self._last_hb = 0.0
        self.last_known_price = self.initial_price or self.avg_price

        if not self.symbol or self.avg_price <= 0 or self.side not in ("Buy","Sell"):
            raise ValueError(f"Bad init: symbol={self.symbol} side={self.side} avg={self.avg_price}")

    # ... (методы _roi_pct, _round_to_tick, _is_better без изменений) ...
    def _roi_pct(self, price: float) -> float:
        if self.side == "Buy":
            pnl = (price / self.avg_price) - 1.0
        else:
            pnl = (self.avg_price / price) - 1.0
        return pnl * 100.0 * self.leverage

    def _round_to_tick(self, price: float) -> float:
        """
        [ФИНАЛЬНАЯ ВЕРСЯ] Корректно округляет цену до шага, требуемого биржей,
        используя стандартный математический подход.
        """
        if self.tick_size <= 0: 
            return price
        
        # Это самый надежный и универсальный способ округления до любого шага цены (tick_size).
        # Он не зависит от "floor" или "ceil" и работает правильно для всех направлений.
        # Пример: price=0.144343, tick_size=0.000001 -> round(144343.95) * 0.000001 = 144344 * 0.000001 = 0.144344
        return round(price / self.tick_size) * self.tick_size

            

    def _is_better(self, cand: float) -> bool:
        if not cand or cand <= 0: return False
        if self.current_stop is None: return True
        return cand > self.current_stop if self.side == "Buy" else cand < self.current_stop
        
    def on_price(self, price: float):
        if price <= 0: return
        self.last_known_price = price
        roi = self._roi_pct(price)

        if not self.trailing_activated and roi >= self.start_roi:
            self.trailing_activated = True
            jprint({"event":"activated","symbol":self.symbol,"roi_pct":roi,"price":price})
            log_err(f"{self.symbol} activated at ROI {roi:.2f}%")

        if self.trailing_activated:
            gap_roi = self.gap_roi_pct
            target_roi = roi - gap_roi
            denom = 1.0 + (target_roi / (100.0 * self.leverage))

            if self.side.lower() == "buy":
                cand = self.avg_price * denom
            else:
                cand = self.avg_price / denom if denom > 1e-9 else None

            if cand is not None:
                ns = self._round_to_tick(cand)
                if self._is_better(ns):
                    prev = self.current_stop
                    self.current_stop = ns
                    jprint({"event":"stop_update","symbol":self.symbol,"stop":self.current_stop,"prev_stop":prev,"reason":"trail"})

    def on_heartbeat(self):
        now = time.time()
        if now - self._last_hb >= self.hb_interval:
            self._last_hb = now
            roi = self._roi_pct(self.last_known_price)
            jprint({"event":"hb","symbol":self.symbol,"activated":self.trailing_activated,
                    "stop":self.current_stop,"price":self.last_known_price,"roi_pct":roi})

# --- НОВАЯ ЛОГИКА ГЛАВНОГО ЦИКЛА ---

async def main_loop(worker: TrailingStopWorker):
    """
    [ИСПРАВЛЕННАЯ ВЕРСИЯ] Читает stdin в ожидании цен и команд,
    используя безопасный `asyncio.wait_for` для избежания гонки состояний.
    """
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)

    if worker.initial_price > 0:
        worker.on_price(worker.initial_price)

    while True:
        try:
            # Пытаемся прочитать строку из stdin с таймаутом в 1 секунду
            line = await asyncio.wait_for(reader.readline(), timeout=1.0)
            
            if not line:
                log_err("Stdin closed, worker shutting down.")
                break
            
            try:
                data = json.loads(line.decode("utf-8", "ignore").strip())
                op = data.get("op")
                if op == "price_update":
                    worker.on_price(f(data.get("price")))
                elif op == "close":
                    jprint({"event": "closed_by_parent", "symbol": worker.symbol})
                    break
            except (json.JSONDecodeError, KeyError):
                log_err(f"Received invalid command: {line.strip()}")

        except asyncio.TimeoutError:
            # Это ожидаемое событие, если нет новых цен. Просто продолжаем.
            pass
        except asyncio.CancelledError:
            log_err("Main loop cancelled.")
            break
        except Exception as e:
            log_err(f"Critical error in main loop: {e}")
            await asyncio.sleep(1) # Пауза в случае непредвиденной ошибки
        
        # Периодически отправляем heartbeat, независимо от того, были ли новые данные
        worker.on_heartbeat()


async def main(params: dict):
    try:
        w = TrailingStopWorker(params)
    except Exception as e:
        log_err(f"Init error: {e}")
        jprint({"event":"init_error","error":str(e)})
        return

    jprint({"event":"init_ok","symbol":w.symbol,"side":w.side,"avg_price":w.avg_price})
    
    await main_loop(w)
    
    jprint({"event":"closed"})
    log_err(f"Worker {w.symbol} done.")

if __name__ == "__main__":
    try:
        params = json.loads(sys.argv[1]) if len(sys.argv)>1 else {}
        asyncio.run(main(params))
    except Exception as e:
        log_err(f"Critical: {e}")
        sys.exit(1)