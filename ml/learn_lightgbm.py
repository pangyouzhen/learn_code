from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import train_test_split

# 增加日期路径，方便保存多个模型结果
today = datetime.today().strftime("%Y%m%d")
path = Path(f"./{today}")
if not path.exists():
    path.mkdir()


iris = load_iris()
data = iris.data
target = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.2, random_state=0
)
# 使用pkl加快加载的速度
X_train.to_pickle(path / "X_train.pkl")
X_test.to_pickle(path / "X_test.pkl")
y_train.to_pickle(path / "y_train.pkl")
y_test.to_pickle(path / "y_test.pkl")

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    # "objective":"multiclass",
    # "metric":"multi_logloss",
    "metric": "rmse",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    # 只显示致命错误
    "verbose": -1,
}

gbm = lgb.train(
    params,
    lgb_train,
    num_boost_round=100,
    valid_sets=lgb_eval,
    early_stopping_rounds=10,
)
x_pred = gbm.predict(X_train, num_iteration=gbm.best_iteration)
print("RMSE:", mean_squared_error(x_pred, X_test) ** 0.5)
gbm.save_model(path / "model.txt")

gbm = lgb.Booster(model_file="model.txt")
y_pred = gbm.predict(X_test)
print("RMSE:", mean_squared_error(y_test, y_pred) ** 0.5)

# 多分类指标
# y_pred = np.argmax(y_pred,axis=1)
# print("F1:",f1_score(y_test,y_pred,average='macro'))

# %matplotlib inline
lgb.plot_importance(gbm)
pd.DataFrame(
    {
        "feature": gbm.feature_name(),
        "importance": gbm.feature_importance(importance_type="gain"),
    }
).sort_values(by="importance", ascending=False)
