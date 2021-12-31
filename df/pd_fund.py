from datetime import datetime
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from pandas.core.groupby import DataFrameGroupBy


# 数据分析还是建议使用jupyter
# 普通的数据分析和具体上系统是不一样的
# 数据的一般流程

# logger 的一些设置


def check_pd_version():
    # 检查pd版本, 防止因为版本不同导致接口不同
    # 适合整体服务启动初始化时检查

    try:
        assert float(pd.__version__[:3]) >= 1.2
    except Exception as e:
        logger.error(f"pd的版本为{pd.__version__},不符合版本")
        raise e


class PreProcess:
    # 进行dataframe的预处理，包括列名称的检查，df的类型转换
    @staticmethod
    def check_df(df: pd.DataFrame, strict_check_columns:bool = False,
                 check_columns: bool = False,
                 column_name: Optional[List[str]] = None,
                 ) -> None:
        # 检查列名是否OK
        if strict_check_columns:
            if df.columns != column_name:
                logger.error("assert的名称和输入的不同，请检查")
        if not strict_check_columns and check_columns:
            if set(df.columns) != set(column_name):
                logger.error("assert的名称和输入的不同，请检查")
        # 检查是是否为空dataframe
        if df.empty:
            logger.error("输入df的为空")
            return 
        if df.shape[0] > 10 ** 7:
            logger.error(f"df的维度是{df.shape}数据的行数过大，请分块输入")
        if df.shape[1] == 0:
            logger.error("输入的空列，请检查")
        if df.index.duplicated is True:
            logger.error("输入的index有重复值")
        logger.info(f"输入的df的维度的时{df.shape}")

    @staticmethod
    def convert_dtypes_na(df: pd.DataFrame) -> pd.DataFrame:
        df = df.convert_dtypes()
        contain_na_colums: List[str] = df.columns[df.isnull().any()].tolist()
        for i in df.select_dtypes(include=np.number).columns:
            if i not in contain_na_colums:
                # Float64Dtype() 在1.2 版本中
                if isinstance(df[i].dtypes, pd.Float64Dtype()):
                    df[i] = df[i].astype(np.float32)
                else:
                    df[i] = df[i].astype(np.int32)
        return df

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        self.check_df(df)
        self.df = self.convert_dtypes_na(df)
        return self.df


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

path = Path("../learn_torch/.data/train.csv")
df = pd.read_csv(path, sep="\t", names=["sent0", "sent1", "label"])
df = df.convert_dtypes()
#  pandas针对大数据进行处理 https://pandas.pydata.org/docs/user_guide/scale.html
#  一般而讲float32保留8位小数，所以对于普通场景都可以将float64转化为float32降低内存
# 数据review
#  注意pandas和numpy的索引选择等不一致
print(df[:5])
# 数据index 检查,index的意义为加快查询速度
assert df.index.has_duplicates is False
# 根据时间选取行
left = pd.DataFrame(
    {
        "a": [1, 2, 3],
        "dt": ["2021/6/11", "2021/07/12", "2021/03/02"],
        "b": [4, 1, np.NAN],
        "c": ["a", "b", "c"],
        "d": [3.0, 4.0, 7.1],
    }
)
ind: pd.DatetimeIndex = pd.DatetimeIndex(["2021-09-09", "2021-09-11", "2021-09-10"])
left.index = ind

right: pd.DataFrame = pd.DataFrame(
    {
        "a": [1, 2, 2, 2],
        "dt": ["2021/6/11", "2021/07/12", "2021/03/02", "2021/09/11"],
        "b": [4, 1, np.NAN, np.NAN],
        "c": ["a", "b", "c", "e"],
        "y": [0, 1, 1, 0],
    }
)
left = left.convert_dtypes()
print(left.dtypes)
dt = datetime(year=2021, month=7, day=1)
left["dt"] = pd.to_datetime(left["dt"])
right["dt"] = pd.to_datetime(right["dt"])
df1 = left[left["dt"] < dt]
# 选择行和列 以后都可以用loc,还有一种选择的方法是iloc
# 直接使用[[]] 选择列这些比较乱
df2: pd.DataFrame = left.loc[left["dt"] < dt, :]
df3: pd.DataFrame = left.loc[left["dt"] < dt, ["a", "b"]]
df4: pd.DataFrame = left.loc[(left["dt"] < dt) & (left["b"] < 4), ["a", "b"]]
df5: pd.Series = right.loc[right["dt"] < dt, "y"]
df5: pd.DataFrame = right.loc[right["c"].isin(["a","b"].squeeze()),:]
df3: pd.DataFrame = df3[~df3.index.duplicated(keep="first")]
# pandas 自动推断并转换类型
for i in df.select_dtypes(include=np.number).columns:
    print(i)
for i in df.select_dtypes(include=np.int64).columns:
    df[i] = df[i].astype(np.int32)
print("-----------------------------------------")
# 这些默认生成的都是series
print(left.dtypes)
print(type(left.dtypes))
# accessors
left["c"] = left["c"].astype("category").cat.codes
print(left)
# print(df1 == df2)
# 选择列,单列可以使用df["b"],多列使用df[["b"]]
# print(df[["b"]])
print("------------------------")
for i in left.columns:
    print(i, left[i].isnull().sum())
# pandas 操作 apply map applymap操作

# pandas 合并表的操作merge, join, concat
# merge 包含了join 的所有操作

df = pd.merge(left, right, left_on="a", right_on="a", how="inner")
print("----------------------数据的描述性统计-------------------")

sk = right.skew().to_frame("偏度")
# print(left.describe())
null_num = right.isnull().sum().to_frame("空值统计")
ku = right.kurt().to_frame("峰度")
df_type = right.dtypes.to_frame("数据类型")
nuniq = right.nunique().to_frame("数据值的个数")
# 堆积柱状图
df = pd.DataFrame({"x": [1, 2, 2, 0, 2], "y": [0, 1, 0, 1, 0]})
a: DataFrameGroupBy = df.groupby(["x", "y"])
for i, v in a:
    pass
df.groupby(["x", "y"]).size().unstack().plot(kind="bar", stacked=True)
plt.show()
plt.close()
print(left.kurt())
df = pd.concat([sk, null_num, ku, df_type, nuniq], axis=1)
print(df)
#  数据清洗，缺失值，异常值，重复值，基本数据分析
#  特征选择  woe 和 iv

np.random.seed(100)

df = pd.DataFrame(
    {
        "grade": np.random.choice(list("ABCD"), size=(20)),
        "pass": np.random.choice([0, 1], size=(20)),
    }
)

feature, target = "grade", "pass"
df_woe_iv = (
    pd.crosstab(df[feature], df[target], normalize="columns")
        .assign(woe=lambda dfx: np.log(dfx[1] / dfx[0]))
        .assign(iv=lambda dfx: np.sum(dfx["woe"] * (dfx[1] - dfx[0])))
)

print(df_woe_iv)

# pandas 常见错误
# ValueError: The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().
# 造成该问题的原因一般是没设置索引,loc通过条件查询,查询出来的是 series序列
# left.loc[left["a"]==1,"c"],可以将这个值打印出来
# left.loc['a',c] 就不会出现这个问题

for i in df.iterrows():
    pass

# 这是一个面板数据，适用于multiIndex,参考股票的数据
df_all = pd.DataFrame(
    {
        "stock_code": ["1", "1", "3"],
        "date": ["2021/6/11", "2021/07/12", "2021/6/11"],
        "value": [4, 1, np.NAN],
        "other": ["a", "b", "c"],
    }
)
df_all = df_all.convert_dtypes()
print(df_all.dtypes)
# 设置双重索引
df_all = df_all.set_index(["date", "stock_code"])
