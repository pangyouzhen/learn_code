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
# reader = pd.read_csv("data/servicelogs", iterator=True)
# # read_csv
# loop = True
# chunkSize = 1000000
# chunks = []
# while loop:
#     try:
#         chunk = reader.get_chunk(chunkSize)
#         chunks.append(chunk)
#     except StopIteration:
#         loop = False
#     print("Iteration is stopped.")
# df = pd.concat(chunks, ignore_index=True)

import numpy as np


def reduce_memo_usage(df: pd.DataFrame, verbose=True):
    """reduce pandas memory usage"""
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage  is: {:.2f} MB".format(start_mem))
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max > np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(
            "Memory usage after optimization is: {:.2f} MB {:.1f}% reduce".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df


if __name__ == "__main__":
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [5.0, 6.0, 7.0, 8.0],
        }
    )
    print(df.info())
    print(reduce_memo_usage(df))
