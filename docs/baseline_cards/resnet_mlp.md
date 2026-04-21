# ResNet-like MLP

- Method name: ResNet-like MLP
- Category: universal
- Applicable data: any prepared dataset with numeric features and explicit split JSON
- Input requirement: shared universal feature builder over canonical prepared format
- Main hyperparameters: `hidden_layers`, `learning_rate`, `batch_size`, `max_iter`, `alpha`, `skip_alpha`
- Fairness note: residual skip is learned only from the same shared input matrix
- Extra prior: none
- Current status: runnable
