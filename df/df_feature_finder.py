from pathlib import Path

import arrow
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
from pandas.core.groupby import DataFrameGroupBy
import numpy as np

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
            logger.info(f"----------类别数过多,请检查------")

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
        un_df: pd.Series = unstack_df.fillna(0).apply(sum, axis=1)
        un_df: pd.DataFrame = un_df.rename('total_sum').to_frame()
        new_df = unstack_df.agg(['idxmax', 'max'], axis=1)
        # 某类别总共有total_sum数，其中idxmax的标签值最多，有max个，占比percent
        final_df = pd.concat([un_df, new_df], axis=1)
        final_df["percent"] = final_df["max"] / final_df["total_sum"]
        logger.info('\n' + final_df.to_string().replace('\n', '\n\t'))
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
