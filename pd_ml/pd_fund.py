import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

# df = pd.read_csv("../full_data/train.csv")
# print(df[:5])
#
# 根据时间选取行
left = pd.DataFrame(
    {
        "a": [1, 2, 3],
        "dt": ["2021/6/11", "2021/07/12", "2021/03/02"],
        "b": [4, 1, np.NAN],
        "c": ["a", "b", "c"]
    }
)

right = pd.DataFrame(
    {
        "a": [1, 2, 3],
        "dt": ["2021/6/11", "2021/07/12", "2021/03/02"],
        "b": [4, 1, np.NAN],
        "c": ["a", "b", "c"]
    }
)
print(left.dtypes)
dt = datetime(year=2021, month=7, day=1)
left["dt"] = pd.to_datetime(left["dt"])
df1 = left[left["dt"] < dt]
#  选择行和列 以后都可以用loc,还有一种选择的方法是iloc
# 直接使用[[]] 选择列这些比较乱
df2 = left.loc[left["dt"] < dt, :]
df3 = left.loc[left["dt"] < dt, ["a", "b"]]
df4 = left.loc[(left["dt"] < dt) & (left["b"] < 4), ["a", "b"]]
# pandas 自动推断并转换类型
left = left.convert_dtypes()
print(left.dtypes)
print(type(left.dtypes))
# 转换成类别的形式
left['c'] = left["c"].astype("category").cat.codes
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
