import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler
from torch.optim.lr_scheduler import (
    LinearLR,                  # Linearly increases/decreases LR over a fixed period.
    CosineAnnealingLR,         # Decreases LR following a cosine curve over a fixed period.
    ReduceLROnPlateau,         # Reduces LR when a monitored metric stops improving.
    OneCycleLR,                # Implements the 1cycle policy (warm-up then cool-down).
    CosineAnnealingWarmRestarts # Cosine annealing with periodic restarts to the initial LR.
)
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
from torch.optim import AdamW



class Trainer:
    def __init__(self, model, tokenizer, args=None, train_ds=None, eval_ds=None, evaluation_data=None):
        self.model = model
        self.tokenizer = tokenizer

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = self.model.to(self.device)

        if args is not None:
            self.args = args
            self.train_ds = train_ds
            self.eval_ds = eval_ds

            self.train_dataloader = DataLoader(self.train_ds, batch_size=self.args.per_device_train_batch_size, shuffle=True)
            self.eval_dataloader = DataLoader(self.eval_ds, batch_size=self.args.per_device_eval_batch_size, shuffle=False)

            self.optimizer = AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)

            self.scheduler_type = self.args.scheduler_type
            self.lr_history = []

            self.early_stopping_patience = self.args.early_stopping_patience if self.args.early_stopping_patience else 3
            self.early_stopping_metric = self.args.early_stopping_metric if self.args.early_stopping_metric else 'val_loss'
            self.best_val_metric = float('inf') if self.early_stopping_metric == 'val_loss' else -float('inf')
            self.epochs_without_improvement = 0

            self.history = {
                "train_loss": [],
                "train_accuracy": [],
                "val_loss": [],
                "val_accuracy": []
            }

        self.evaluation_data = evaluation_data
    def train(self):

        scheduler = self._init_lr_scheduler()
        accumulation_steps = self.args.gradient_accumulation or 1  # default to 1 if None

        self.model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        for epoch in range(self.args.num_train_epochs):
            print(f"{'='*15} Epoch {epoch+1} {'='*15}")
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1} [Train]")
            for batch_idx, batch in enumerate(progress_bar):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                output = self.model(**batch)
                
                loss = self.args.loss_fn(output.logits, batch["labels"])
                raw_loss = loss  # keep original for logging

                if accumulation_steps > 1:
                    loss = loss / accumulation_steps  # scale for backward

                loss.backward()

                is_accumulation_step = (batch_idx + 1) % accumulation_steps == 0
                is_last_batch = (batch_idx + 1) == len(self.train_dataloader)

                if is_accumulation_step or is_last_batch:
                    self.optimizer.step()

                    if self.scheduler_type != "reduce_on_plateau":
                        scheduler.step()

                    self.optimizer.zero_grad()

                train_loss += raw_loss.item()  # accumulate the *unscaled* loss

                current_lr = self.optimizer.param_groups[0]['lr']
                self.lr_history.append(current_lr)

                logits = output.logits
                predictions = torch.argmax(logits, dim=-1)
                labels = batch["labels"]

                train_preds.extend(predictions.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())

                progress_bar.set_postfix({"training loss": raw_loss.item(), "lr": current_lr})


            train_loss = train_loss / len(self.train_dataloader)
            train_accuracy = accuracy_score(train_labels, train_preds)

            print(f"Train Loss: {train_loss}, Train Accuracy: {train_accuracy}")

            # Evaluation Step
            self.model.eval()
            val_loss = 0
            val_preds = []
            val_labels = []

            with torch.no_grad():
                progress_bar = tqdm(self.eval_dataloader, desc=f"Epoch {epoch+1} [Eval]")
                for batch in progress_bar:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    output = self.model(**batch)
                    loss = self.args.loss_fn(output.logits, batch["labels"])

                    val_loss += loss.item()

                    logits = output.logits
                    predictions = torch.argmax(logits, dim=-1)
                    labels = batch["labels"]

                    val_preds.extend(predictions.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

                    progress_bar.set_postfix({"validation loss": loss.item()})
            
            val_loss = val_loss / len(self.eval_dataloader)
            val_accuracy = accuracy_score(val_labels, val_preds)

            self.history["train_loss"].append(train_loss)
            self.history["train_accuracy"].append(train_accuracy)
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_accuracy)

            print(f"Val Loss: {val_loss}, Val Accuracy: {val_accuracy}")

            if self.scheduler_type == "reduce_on_plateau":
                scheduler.step(val_loss)


            # === EARLY STOPPING CHECK ===
            current_metric = val_loss if self.early_stopping_metric == 'val_loss' else val_accuracy

            improvement = False
            if self.early_stopping_metric == 'val_loss':
                if current_metric < self.best_val_metric:
                    improvement = True
                    self.best_val_metric = current_metric
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1
            elif self.early_stopping_metric == 'val_accuracy':
                if current_metric > self.best_val_metric:
                    improvement = True
                    self.best_val_metric = current_metric
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1

            print(f"[EarlyStopping] {self.early_stopping_metric}: {current_metric:.4f}, "
                  f"Best: {self.best_val_metric:.4f}, "
                  f"Epochs without improvement: {self.epochs_without_improvement}")

            if self.early_stopping_patience is not None and self.epochs_without_improvement >= self.early_stopping_patience:
                print(f"Early stopping triggered! No improvement in {self.early_stopping_patience} consecutive epochs.")
                break



        return self.history

    def _init_lr_scheduler(self):
        # Prepare scheduler based on the selected type
        num_training_steps = self.args.num_train_epochs * len(self.train_dataloader)
        num_warmup_steps = int(self.args.warmup_ratio * num_training_steps)
        
        if self.scheduler_type == "linear":
            scheduler = get_scheduler(
                name="linear",
                optimizer=self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        elif self.scheduler_type == "cosine":
            scheduler = get_scheduler(
                name="cosine",
                optimizer=self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        elif self.scheduler_type == "cosine_annealing":
            scheduler = CosineAnnealingLR(
                optimizer=self.optimizer,
                T_max=num_training_steps
            )
        elif self.scheduler_type == "one_cycle":
            scheduler = OneCycleLR(
                optimizer=self.optimizer,
                max_lr=self.args.learning_rate * 10,
                total_steps=num_training_steps
            )
        elif self.scheduler_type == "cosine_warm_restarts":
            scheduler = CosineAnnealingWarmRestarts(
                optimizer=self.optimizer,
                T_0=len(self.train_dataloader)  # Restart every epoch
            )
        elif self.scheduler_type == "reduce_on_plateau":
            scheduler = ReduceLROnPlateau(
                optimizer=self.optimizer,
                mode='min',
                factor=0.5,
                patience=2,
                verbose=True
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")

        return scheduler

    def predict(self, texts):
        self.model.eval()
        predictions = []

        for text in texts:
            inputs = self.tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                output = self.model(**inputs)
                logits = output.logits
                pred = torch.argmax(logits, dim=-1)

            label_map = {0: "Positive", 1: "Neutral", 2: "Negative"}
            predictions.append({
                "text": text,
                "sentiment": label_map[pred.item()]
            })
            
        return predictions

    def _evaluate(self):
        if self.evaluation_data is None:
            return "THERE IS NO DATA TO EVALUATE. PLEASE PASS evaluation_data=TEST_DATALOADER"
        
        self.model.eval()
        y_pred = []
        y_true = []

        with torch.no_grad():
            for batch in tqdm(self.evaluation_data, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)
                labels = batch["labels"]

                y_pred.extend(preds.cpu().numpy())
                y_true.extend(labels.cpu().numpy())

        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred)

        return {
            "Accuracy": accuracy,
            "F1-Score": f1,
            "Confusion_matrix": cm,
            "Classification_report": report
        }

    def print_evaluation_report(self):
        results = self._evaluate()
        for e, v in results.items():
            if e != "Confusion_matrix": print(f"{e}: {v}")
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(results["Confusion_matrix"], annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.show()

    def plot_history(self):
        epochs = range(1, self.args.num_train_epochs+1)
        plt.figure(figsize=(12, 5))

        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], 'b-', label='Training Loss')
        plt.plot(self.history['val_loss'], 'r-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_accuracy'], 'b-', label='Training Accuracy')
        plt.plot(self.history['val_accuracy'], 'r-', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def plot_lr_history(self):
        plt.figure(figsize=(12, 5))
        plt.plot(self.lr_history)
        plt.title('Learning Rate History')
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.show()

    def visualize_lr_schedulers(self, steps_per_epoch=205):
        base_lr = self.args.learning_rate
        total_steps = self.args.num_train_epochs * steps_per_epoch
        
        # Create a dummy model parameter to update
        dummy_param = torch.nn.Parameter(torch.zeros(1))
        
        # Define schedulers to compare
        schedulers = {
            # HuggingFace schedulers
            'HF Linear': get_scheduler(
                name="linear",
                optimizer=AdamW([dummy_param], lr=base_lr),
                num_warmup_steps=int(self.args.warmup_ratio * total_steps),
                num_training_steps=total_steps
            ),
            'HF Cosine': get_scheduler(
                name="cosine",
                optimizer=AdamW([dummy_param], lr=base_lr),
                num_warmup_steps=int(self.args.warmup_ratio * total_steps),
                num_training_steps=total_steps
            ),
            
            # PyTorch schedulers
            'Cosine Annealing': CosineAnnealingLR(
                optimizer=AdamW([dummy_param], lr=base_lr),
                T_max=total_steps
            ),
            'One Cycle': OneCycleLR(
                optimizer=AdamW([dummy_param], lr=base_lr),
                max_lr=base_lr * 10,
                total_steps=total_steps
            ),
            'Cosine Warm Restarts': CosineAnnealingWarmRestarts(
                optimizer=AdamW([dummy_param], lr=base_lr),
                T_0=steps_per_epoch  # Restart every epoch
            ),
            'ReduceLROnPlateau': ReduceLROnPlateau(
                optimizer=AdamW([dummy_param], lr=base_lr),
                mode='min',
                factor=0.5,
                patience=2,
                verbose=True
            )
        }
        
        # Track learning rates
        lr_history = {name: [] for name in schedulers}
        
        # Simulate steps
        for step in range(total_steps):
            for name, scheduler in schedulers.items():
                # Get current learning rate
                current_lr = scheduler.optimizer.param_groups[0]['lr']
                lr_history[name].append(current_lr)
                
                # Update scheduler
                if name == 'ReduceLROnPlateau':
                    # This scheduler needs validation loss
                    # For visualization, we'll use a dummy value
                    scheduler.step(1.0)
                else:
                    scheduler.step()
        
        # Visualize learning rates
        plt.figure(figsize=(12, 6))
        
        for name, lrs in lr_history.items():
            plt.plot(lrs, label=name)
        
        plt.xlabel('Training Steps')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedulers Comparison')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add epoch boundaries
        for i in range(1, self.args.num_train_epochs):
            plt.axvline(x=i * steps_per_epoch, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()

    def save_model(self, output_dir, file_name="pytorch_model.bin"):
        import os
        os.makedirs(output_dir, exist_ok=True)
    
        if hasattr(self.model, "save_pretrained"):  # Hugging Face model
            self.model.save_pretrained(output_dir)
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)
            return f"Model and tokenizer saved to {output_dir}"
        else:  # PyTorch native model
            torch.save(self.model.state_dict(), os.path.join(output_dir, file_name))
            return f"PyTorch model state_dict saved to {output_dir}/{file_name}"