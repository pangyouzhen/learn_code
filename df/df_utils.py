from loguru import logger
import arrow
import pandas as pd

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
# 大文件分块读取，最好先用linux  wc -l 检查多少行，
reader = pd.read_csv('data/servicelogs', iterator=True)
# read_csv
loop = True
chunkSize = 1000000
chunks = []
while loop:
    try:
        chunk = reader.get_chunk(chunkSize)
        chunks.append(chunk)
    except StopIteration:
        loop = False
    print("Iteration is stopped.")
df = pd.concat(chunks, ignore_index=True)