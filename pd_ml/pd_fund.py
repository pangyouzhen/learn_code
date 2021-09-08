import numpy as np
import pandas as pd
from datetime import datetime
# import matplotlib.pyplot as plt
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
ind: pd.DatetimeIndex = pd.DatetimeIndex(['2021-09-09', '2021-09-11', '2021-09-10'])
left.index = ind
# 一般而言索引是不能重复的,
# 索引的重复值 判断
assert len(set(left.index.duplicated())) == 1
dtype_dict = {
    "dt": np.datetime64,
    "a": int,
}

right: pd.DataFrame = pd.DataFrame(
    {
        "a": [1, 2, 2, 2],
        "dt": ["2021/6/11", "2021/07/12", "2021/03/02", "2021/09/11"],
        "b": [4, 1, np.NAN, np.NAN],
        "c": ["a", "b", "c", "e"],
        "y": [0, 1, 1, 0]
    }
)
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
df3 = df3[~df3.index.duplicated(keep='first')]
# pandas 自动推断并转换类型
print("--------------类型推断-----------------------------")
# left = left.convert_dtypes()
for i in left.select_dtypes(exclude=np.number).columns:
    print(i)
print("-----------------------------------------")
# 这些默认生成的都是series
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
print("----------------------数据的描述性统计-------------------")

sk = right.skew().to_frame("偏度")
# print(left.describe())
null_num = right.isnull().sum().to_frame("空值统计")
ku = right.kurt().to_frame("峰度")
df_type = right.dtypes.to_frame("数据类型")
nuniq = right.nunique().to_frame("数据值的个数")
# 堆积柱状图
# df = pd.DataFrame(
#     {"x": [1, 2, 2, 0, 2],
#      "y": [0, 1, 0, 1, 0]}
# )
# df.groupby(["x", "y"]).size().unstack().plot(kind='bar', stacked=True)
# plt.show()
# plt.close()
# print(left.kurt())
df = pd.concat([sk, null_num, ku, df_type, nuniq], axis=1)
print(df)
#  数据清洗，缺失值，异常值，重复值，基本数据分析
#  特征选择  woe 和 iv
import numpy as np
import pandas as pd

np.random.seed(100)

df = pd.DataFrame({'grade': np.random.choice(list('ABCD'), size=(20)),
                   'pass': np.random.choice([0, 1], size=(20))
                   })

feature, target = 'grade', 'pass'
df_woe_iv = (pd.crosstab(df[feature], df[target],
                         normalize='columns')
             .assign(woe=lambda dfx: np.log(dfx[1] / dfx[0]))
             .assign(iv=lambda dfx: np.sum(dfx['woe'] *
                                           (dfx[1] - dfx[0]))))

print(df_woe_iv)

# pandas 常见错误
# ValueError: The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().
# 造成该问题的原因一般是没设置索引,loc通过条件查询,查询出来的是 series序列
# left.loc[left["a"]==1,"c"],可以将这个值打印出来
# left.loc['a',c] 就不会出现这个问题

for i in df.iterrows():
    pass
