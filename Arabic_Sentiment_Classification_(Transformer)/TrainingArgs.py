class TrainerArgs:
    def __init__(
            self,
            per_device_train_batch_size,
            per_device_eval_batch_size,
            learning_rate,
            warmup_ratio,
            weight_decay,
            num_train_epochs,
            loss_fn,
            scheduler_type="linear",
            gradient_accumulation=None,
            early_stopping_patience=2,
            early_stopping_metric="val_loss",
    ):
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.num_train_epochs = num_train_epochs
        self.loss_fn = loss_fn
        self.scheduler_type = scheduler_type
        self.gradient_accumulation = gradient_accumulation
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_metric = early_stopping_metric

    def get_args(self):
        return f"""
        per_device_train_batch_size: {self.per_device_train_batch_size}
        per_device_eval_batch_size: {self.per_device_eval_batch_size}
        learning_rate: {self.learning_rate}
        warmup_ratio: {self.warmup_ratio}
        scheduler_type: {self.scheduler_type}
        weight_decay: {self.weight_decay}
        num_train_epochs: {self.num_train_epochs}
        loss_fn: {self.loss_fn}
        gradient_accumulation: {self.gradient_accumulation}
        early_stopping_patience: {self.early_stopping_patience}
        early_stopping_metric: {self.early_stopping_metric}
        """