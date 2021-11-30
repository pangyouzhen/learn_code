```sequence
gbdt_instance->BaseGradientBoosting:fit
BaseGradientBoosting->BaseGradientBoosting:_fit_stages
BaseGradientBoosting->BaseGradientBoosting:_fit_stage
BaseGradientBoosting->MultinomialDeviance:negative_gradient
BaseGradientBoosting->DecisionTreeRegressor:fit
```