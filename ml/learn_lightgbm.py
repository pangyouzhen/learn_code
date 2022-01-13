import lightgbm as lgb
from sklearn.datasets import load_iris
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

iris = load_iris()
data = iris.data
target = iris.target

X_train,X_test,y_train,y_test = train_test_split(data,target,test_size=0.2,random_state=0)
print(X_train.shape)
print(X_test.shape)

lgb_train = lgb.Dataset(X_train,y_train)
lgb_eval = lgb.Dataset(X_test,y_test,reference=lgb_train)

params = {
    "task":"train",
    "boosting_type":"gbdt",
    "objective":"regression",
    "metric":"rmse",
    "num_leaves":31,
    "learning_rate":0.05,
    "feature_fraction":0.9,
    "bagging_fraction":0.8,
    "bagging_freq":5,
    "verbose":0
}

gbm = lgb.train(params,lgb_train,num_boost_round=100,valid_sets=lgb_eval,early_stopping_rounds=10)
gbm.save_model("model.txt")

gbm = lgb.Booster(model_file="model.txt")
y_pred = gbm.predict(X_test)
print("RMSE:",mean_squared_error(y_test,y_pred)**0.5)
