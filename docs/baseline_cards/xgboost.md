# XGBoost

- Method name: XGBoost
- Category: universal
- Applicable data: any prepared dataset with numeric features and explicit split JSON
- Input requirement: shared universal feature builder over canonical prepared format
- Main hyperparameters: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `n_jobs`
- Fairness note: uses the same split, targets, and metrics interface as the other universal baselines
- Extra prior: none
- Current status: runnable when `xgboost` is installed
