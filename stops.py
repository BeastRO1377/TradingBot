# stops.py
import asyncio, logging
logger = logging.getLogger(__name__)

async def monitor(bot, interval: int = 5):
    """
    Каждые <interval> секунд вызывает bot.check_stop_conditions().
    Работает, пока bot.active == True.
    """
    logger.info(f"[stops.monitor] запущен для user_id={bot.user_id}, interval={interval}s")
    try:
        while bot.active:
            try:
                await bot.check_stop_conditions()
            except Exception as e:
                logger.error(f"[stops.monitor] ошибка: {e}")
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        logger.info(f"[stops.monitor] отменён (user_id={bot.user_id})")