import re

import pandas as pd
from sklearn.metrics import confusion_matrix
# 机器学习，深度学习的效果评估指标, sklearn的效果评估都在 metrics中\
# 常见的评估指标
# 准确率，召回率，f1，混淆矩阵, 在sklearn中输入都是 先输入正确值，再输入预测值 y_true, y_pred
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

df = pd.read_excel("./data/test.xlsx", encoding="utf-8")
# 对列名进行重命名
df.columns = ["user", "std", "answer"]
# 查看df的每列的数据类型

print(df.dtypes)
punction = re.compile("[\\\\/`~!]")


#
#  抽取user和std 完全相同的
def user_std_match(x):
    # 这里需要加上str将原始的数据进行转换，否则可能出现用户输入1这种报错
    user = punction.sub("", str(x["user"]))
    std = punction.sub("", str(x["std"]))
    if user == std:
        return 1
    return 0


# 针对整个dataframe 进行apply，别忘记添加参数axis=1
df["user_std_match"] = df.apply(lambda x: user_std_match(x), axis=1)
# 数据抽取
df = df[df["user_std_match"] == 0]

# 对某些类进行重新赋值，换成apply 函数比较好
# df.loc[df["user_std_match"] == 0, "resign"] = 1
print(df.shape)
# 进行混淆矩阵的计算，混淆矩阵主要来判断分类的效果，分类效果中经常用到
y_true = df["true"].tolist()
y_pred = df["pred"].tolist()
confusion_matrix_ = confusion_matrix(y_true, y_pred)
print(confusion_matrix_)
print(f1_score(y_true, y_pred))
print(recall_score(y_true, y_pred))
print(precision_score(y_true, y_pred))
