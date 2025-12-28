import subprocess
from typing import Any

import pytorch_lightning as pl
import torch
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall
from transformers import AutoModelForSequenceClassification


class MoneyTextClassifier(pl.LightningModule):
    def __init__(self, model_name: str, lr: float):
        super().__init__()
        self.save_hyperparameters()  # model_name, lr

        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.model.train()

        self.f1 = BinaryF1Score()
        self.precision = BinaryPrecision()
        self.recall = BinaryRecall()

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def training_step(self, batch: dict[str, Any], batch_idx: int):
        out = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"].long(),
        )
        self.log("train_loss", out.loss, prog_bar=True)
        return out.loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int):
        out = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"].long(),
        )
        preds = torch.argmax(out.logits, dim=1)
        y = batch["labels"].long()

        self.log("val_loss", out.loss, prog_bar=True)
        self.log("val_f1", self.f1(preds, y), prog_bar=True)
        self.log("val_precision", self.precision(preds, y))
        self.log("val_recall", self.recall(preds, y))

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

    def on_fit_start(self):
        commit = None
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            pass

        if commit and self.logger is not None:
            try:
                self.logger.log_hyperparams({"git_commit": commit})
            except Exception:
                pass
