import fasttext
from sklearn.model_selection import train_test_split
import jieba
import pandas as pd

#  后面可以将jieba替换成fast jieba

df = pd.read_csv("../full_data/train.csv")
df = df[["Phrase", "Sentiment"]]
jieba.load_userdict("./")
#  下面dict 仅作示例，原始中可以通过 str 相加得到
label_dict = {
    "1": "__label__1",
    "2": "__label__2",
    "3": "__label__3",
    "4": "__label__4"
}
df["label"] = df["Sentiment"].map(label_dict)
#  仅对于中文的做jieba处理
df["jieba"] = df["Phrase"].apply(lambda x: " ".join(list(jieba.cut(str(x)))))
x, y = df["jieba"], df["label"]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)
train_data.to_csv('./full_data/fasttext_train.txt', encoding="utf-8", sep=" ", index=False)
test_data.to_csv('./full_data/fasttext_test.txt', encoding="utf-8", sep=" ", index=False)

classifier = fasttext.supervised("./full_data/fasttext_train.txt", "./full_data/fasttext_train.model",
                                 label_prefix="__label__",
                                 lr=0.1,
                                 lr_update_rate=5,
                                 epoch=30, dim=128, ws=8)

test_data["predict"] = test_data["jieba"].apply(lambda x: classifier.predict_proba([x], k=3))
test_data["predict_label"] = test_data["predict"].apply(lambda x: list(map(str, x[0][0]))[0])
# 下面可以直接用sklearn 的混淆矩阵进行计算
