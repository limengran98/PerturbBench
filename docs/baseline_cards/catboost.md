# CatBoost

- Method name: CatBoost
- Category: universal
- Applicable data: any prepared dataset with numeric features and explicit split JSON
- Input requirement: shared universal feature builder over canonical prepared format
- Main hyperparameters: `iterations`, `depth`, `learning_rate`
- Fairness note: no extra side information is added
- Extra prior: none
- Current status: dependency-gated; wrapper exists but current environment does not have `catboost`
