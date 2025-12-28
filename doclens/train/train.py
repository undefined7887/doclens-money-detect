from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, MLFlowLogger

from doclens.data.data_module import DataModule
from doclens.train.money_text_classifier import MoneyTextClassifier


def train(
    data_dir: Path,
    splits: dict[str, str],
    text_col: str,
    target_col: str,
    model_name: str,
    lr: float,
    epochs: int,
    accelerator: str,
    devices: str,
    log_every_n_steps: int,
    log_dir: Path,
    checkpoint_dir: Path,
    checkpoint_metric: str,
    checkpoint_mode: str,
    ml_flow_tracking_url: str,
    ml_flow_experiment_name: str,
):
    data_module = DataModule(
        data_dir=data_dir,
        splits=splits,
        model_name=model_name,
        text_col=text_col,
        target_col=target_col,
    )

    model = MoneyTextClassifier(model_name=model_name, lr=lr)

    callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best",
        monitor=checkpoint_metric,
        mode=checkpoint_mode,
        save_top_k=1,
    )

    if ml_flow_tracking_url:
        logger = MLFlowLogger(
            tracking_uri=ml_flow_tracking_url, experiment_name=ml_flow_experiment_name
        )
    else:
        logger = CSVLogger(save_dir=log_dir)

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=log_every_n_steps,
        logger=logger,
        callbacks=[callback],
    )

    trainer.fit(model, datamodule=data_module)

    print(f"best model path: {callback.best_model_path}")
