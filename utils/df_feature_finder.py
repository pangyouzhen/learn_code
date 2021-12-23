from pathlib import Path

import arrow
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from pandas.core.groupby import DataFrameGroupBy

path = Path(__file__).stem
run_date = arrow.now().format(f"YYYY_MM_DD_HH_mm")
logger.add(f"./{run_date}_{path}.log")


# versions = f"{sys.version_info[0]}.{sys.version_info[1]}"
def run_time(func):
    def wrapper(*args, **kwargs):
        start_time: arrow.Arrow = arrow.now()
        func(*args, **kwargs)
        end_time: arrow.Arrow = arrow.now()
        last_time = end_time - start_time
        logger.info(f"{func.__name__} 运行耗时 {last_time.seconds}秒")
        return

    return wrapper


class DataFrameFinder:
    def __init__(self, df: pd.DataFrame, label: str, show_jpg=True):
        logger.info(f"df.shape = {df.shape}")
        logger.info(f"label = {label}")
        self.df = df
        self.label = label
        self.show_jpg = show_jpg
        self.check_df()

    @run_time
    def check_df(self):
        """
        检查df，检查分类的类别是否过多
        """
        label_value_counts: pd.Series = self.df[self.label].value_counts()
        logger.info(f"label的类别数有{label_value_counts.shape[0]}")
        if label_value_counts.shape[0] > 20 or (label_value_counts.shape[0] / self.df.shape[0] > 0.8):
            logger.info(f"类别数过多,请检查------")

    @run_time
    def run(self):
        cols = self.df.columns
        for col in cols:
            if col != self.label:
                self.single(col)
        plt.close()

    @run_time
    def single(self, single_col: str):
        logger.info(f"对{single_col}列进行计算")
        group_df: DataFrameGroupBy = self.df.groupby([self.label, single_col])
        unstack_df: pd.DataFrame = group_df.size().unstack()
        idx_df: pd.Series = unstack_df.idxmax(axis=1)
        idx_df: pd.DataFrame = idx_df.rename(f"最大值所在的{single_col}列").to_frame()
        m_df: pd.Series = unstack_df.max(axis=1)
        m_df: pd.DataFrame = m_df.rename("最大值").to_frame()
        max_df = pd.concat([idx_df, m_df], axis=1)
        # 占类别总数的百分比
        max_df["占比"] = max_df["最大值"] / (self.df.shape[0])
        logger.info('\n' + max_df.to_string().replace('\n', '\n\t'))
        unstack_df.plot(kind="bar", stacked=True)
        plt.show()

    def __call__(self, *args, **kwargs):
        return self.run()


if __name__ == '__main__':
    df_all = pd.DataFrame(
        {
            "stock_code": ["1", "1", "3"],
            "date": ["2021/6/11", "2021/07/12", "2021/6/11"],
            "value": [4, 1, np.NAN],
            "other": ["a", "a", "c"],
        }
    )
    t = DataFrameFinder(df_all, "other")
    t()
