from pathlib import Path

import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        model_name: str,
        data_dir: Path,
        splits: dict[str, str],
        text_col: str,
        target_col: str,
        batch_size: int = 8,
        max_length: int = 256,
        num_workers: int = 2,
    ):
        super().__init__()

        # Removing empty splits
        splits = {k: v for k, v in splits.items() if v}

        required_splits = {"train", "validation", "test"}
        missing_splits = required_splits - set(splits)
        if missing_splits:
            raise ValueError(f"Missing required splits: {sorted(missing_splits)}")

        self.model_name = model_name
        self.data_dir = Path(data_dir)
        self.splits = splits
        self.text_col = text_col
        self.target_col = target_col

        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers

        self.ds = None
        self.tokenizer = None

    def setup(self, stage: str | None = None):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)

        files = {
            split: str(self.data_dir / f"{file}.parquet") for split, file in self.splits.items()
        }
        self.ds = load_dataset("parquet", data_files=files)

        cols = self.ds["train"].column_names

        def tokenize(batch):
            enc = self.tokenizer(
                batch[self.text_col],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
            )

            # Using target_col as labels
            enc["labels"] = batch[self.target_col]

            return enc

        self.ds = self.ds.map(tokenize, batched=True, remove_columns=cols)
        self.ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    def _loader(self, split: str, shuffle: bool) -> DataLoader:
        return DataLoader(
            self.ds[split],
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=(self.num_workers > 0),
        )

    def train_dataloader(self):
        return self._loader("train", shuffle=True)

    def val_dataloader(self):
        return self._loader("validation", shuffle=False)

    def test_dataloader(self):
        return self._loader("test", shuffle=False)
