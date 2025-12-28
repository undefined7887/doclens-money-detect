from pathlib import Path

import hydra
from omegaconf import DictConfig

from doclens.data.download import download_dataset
from doclens.data.process import process_dataset
from doclens.train.train import train


@hydra.main(config_path="../config", config_name="download", version_base=None)
def cmd_download(config: DictConfig):
    download_dataset(
        name=config.dataset.name,
        subset=config.dataset.subset,
        splits={v for k, v in config.dataset.splits.items() if v},
        out_dir=Path(config.out_dir),
    )


@hydra.main(config_path="../config", config_name="process", version_base=None)
def cmd_process(config: DictConfig):
    process_dataset(
        splits={v for k, v in config.dataset.splits.items() if v},
        label_col=config.dataset.label_col,
        target_col=config.dataset.target_col,
        money_labels=config.dataset.money_labels,
        raw_dir=Path(config.raw_dir),
        out_dir=Path(config.out_dir),
    )


@hydra.main(config_path="../config", config_name="train", version_base=None)
def cmd_train(config: DictConfig):
    train(
        model_name=config.model.name,
        data_dir=Path(config.data_dir),
        splits=config.dataset.splits,
        text_col=config.dataset.text_col,
        target_col=config.dataset.target_col,
        lr=config.model.lr,
        epochs=config.model.epochs,
        accelerator=config.model.accelerator,
        devices=config.model.devices,
        log_every_n_steps=config.model.log_every_n_steps,
        log_dir=Path(config.log_dir),
        checkpoint_dir=Path(config.checkpoint.dir),
        checkpoint_metric=config.checkpoint.metric,
        checkpoint_mode=config.checkpoint.mode,
        ml_flow_tracking_url=config.ml_flow.tracking_uri,
        ml_flow_experiment_name=config.ml_flow.experiment_name,
    )
