import math
# https://zhuanlan.zhihu.com/p/127032260
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

plt.style.use('fivethirtyeight')
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 28, 18
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
rcParams['axes.unicode_minus'] = False
df = pd.read_csv("./电网现金流数据.csv", index_col=0)
cols = df.columns

all_t = [df[[i]] for i in cols]
fig, axes = plt.subplots(3, 3, figsize=(30, 20))
for t, ax in zip(all_t, axes.flatten()):
    col_name = list(t.columns)[0]
    t.plot(y=col_name, ax=ax, title=col_name, fontsize=25)
    ax.set_xlabel('时间')
    ax.set_ylabel(col_name)
    ax.xaxis.label.set_size(23)
    ax.yaxis.label.set_size(23)
plt.show()

# business_income = df[["business_income"]]
df["business_income"] = df["business_income"].apply(lambda x: math.log(x))
business_income = df[["business_income"]]
# # sale_elec_income = df[["sale_elec_income"]]
# # df["sale_elec_income"] = df["sale_elec_income"].apply(lambda x: math.log(x))
# # plt.subplot(211)
# plt.tick_params(axis='x', labelsize=8)
# plt.plot(business_income, label=u'business_income')
# plt.legend(loc='best')
# plt.title("business_income")
# # plt.subplot(212)
# # plt.plot(sale_elec_income, label=u'sale_elec_income')
# # plt.legend(loc='best')
# plt.show()

#
# df["sale_elec_income"] = df["sale_elec_income"].apply(lambda x: math.log(x))
# sale_elec_income = df[["sale_elec_income"]]
# plt.subplot(211)
# plt.tick_params(axis='x', labelsize=8)
# plt.plot(business_income, label=u'business_income')
# plt.legend(loc='best')
# plt.title("business_income")
# plt.subplot(212)
# plt.tick_params(axis='x', labelsize=8)
# plt.plot(sale_elec_income, label=u'sale_elec_income')
# plt.title("sale_elec_income")
# plt.legend(loc='best')
# plt.show()


# 分解趋势、季节、随机效应
business_income.index = pd.to_datetime(business_income.index)
business_income_series = business_income.squeeze()
decomposition = seasonal_decompose(business_income_series)
trend = decomposition.trend  # 趋势效应

seasonal = decomposition.seasonal  # 季节效应

residual = decomposition.resid  # 随机效应
plt.subplot(411)
plt.plot(business_income, label=u'原始数据')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label=u'趋势')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal, label=u'季节性')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label=u'残差')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
