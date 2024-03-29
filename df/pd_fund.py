from datetime import datetime
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from pandas.core.groupby import DataFrameGroupBy

# pd版本检查，文件夹创建等
today = datetime.today().strftime("%Y-%m-%d")
path = Path(f"{today}")
if not path.exists():
    path.mkdir()

# 读取大数据文件最好使用不要用csv格式，可以使用pickle等格式，csv格式读取和存储比较慢
df = pd.read_csv(path, sep="\t", names=["sent0", "sent1", "label"])
df = df.convert_dtypes()
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
df5: pd.DataFrame = right.loc[right["c"].isin(["a", "b"].squeeze()), :]
df3: pd.DataFrame = df3[~df3.index.duplicated(keep="first")]
df[col].cat.ordered()
# pandas 重新赋值
right.loc[right["dt"] < dt, "y"] = 10
# pandas 自动推断并转换类型

for i in df.select_dtypes(include=np.number).columns:
    print(i)
for i in df.select_dtypes(include=np.int64).columns:
    df[i] = df[i].astype(np.int32)

int64_cols = df.select_dtypes(include=np.int64).columns
for i in set(df.dtypes):
    print(i)
    print(sorted(list(df.dtypes[df.dtypes == i].index)))
df[int64_cols] = df[int64_cols].astype(np.int32)
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
        "pass": np.random.choice([0, 1, 3, 5, np.NAN], size=(20)),
    }
)
# cross table 不会对np.NaN进行统计, cross 的含义是 交叉，是十字
feature, target = "grade", "pass"
df_woe_iv = (
    pd.crosstab(df[feature], df[target], normalize="columns")
        #  这里的0，1 是target中的值的分类
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

df2 = pd.DataFrame(
    {
        "date1": ["2021/1/6", "2022/03/04", "2023/04/05"],
        "date2": ["2021/6/11", "2021/07/12", np.NAN],
        "value": [4, 1, np.NAN],
    }
)
df2["date1"] = pd.to_datetime(df2["date1"])
df2["date2"] = pd.to_datetime(df2["date2"])
# to_datetime和保存的命令类似，以后不如直接都用 astype
df['time'] = df['time'].astype('datetime64[ns]')
df2["interval"] = df2["date2"] - df2["date1"]
df2["time_lag"] = df2["interval"].dt.ceil("D").dt.days

df3 = pd.DataFrame(
    {
        "num1": list(range(100)),
        "num2": list(range(100, 200))
    }
)
# 针对为NaN的也是ok的，NaN不会进行判定
df3["num_bin"] = pd.cut(df3["num1"], bins=[float("-inf"), 10, 30, 60, float("inf")])


from pandas_profiling import ProfileReport

profile = ProfileReport(df3, title="Pandas Profiling Report",minimal=True)
profile.to_file("profile_report.html")