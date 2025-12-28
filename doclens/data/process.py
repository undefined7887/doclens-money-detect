from pathlib import Path

from datasets import load_dataset


def process_dataset(
    splits: set[str],
    label_col: str,
    target_col: str,
    money_labels: set[str],
    raw_dir: Path,
    out_dir: Path,
):
    files = {split: str(raw_dir / f"{split}.parquet") for split in splits}

    ds = load_dataset("parquet", data_files=files)

    def is_money(batch):
        labels = batch[label_col]
        return {target_col: [1 if label in money_labels else 0 for label in labels]}

    for split in splits:
        # Finding labels that are related to money
        d = ds[split].map(is_money, batched=True, batch_size=1000)

        # Dropping the label column
        d = d.remove_columns(label_col)

        d.to_parquet(out_dir / f"{split}.parquet")
