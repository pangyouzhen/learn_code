from loguru import logger
import arrow


class DfUtils:
    def __init__(self):
        pass

    @staticmethod
    def run_time(func):
        def wrapper(*args, **kwargs):
            start_time: arrow.Arrow = arrow.now()
            func(*args, **kwargs)
            end_time: arrow.Arrow = arrow.now()
            last_time = end_time - start_time
            logger.info(f"{func.__name__} 运行耗时 {last_time.seconds}秒")
            return

        return wrapper
