```sequence
gbdt_instance->BaseGradientBoosting:fit
BaseGradientBoosting->BaseGradientBoosting:_fit_stages
BaseGradientBoosting->BaseGradientBoosting:_fit_stage
BaseGradientBoosting->DecisionTreeRegressor:fit
DecisionTreeRegressor->DepthFirstTreeBuilder:_build_tree
DepthFirstTreeBuilder->BestFirstTreeBuilder:_build_tree
```