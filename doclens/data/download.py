from pathlib import Path

from datasets import load_dataset


def download_dataset(name: str, subset: str, splits: set[str], out_dir: Path):
    ds = load_dataset(name, subset)

    for split in splits:
        split_path = out_dir / f"{split}.parquet"

        if split_path.exists():
            split_path.unlink()

        ds[split].to_parquet(split_path)
