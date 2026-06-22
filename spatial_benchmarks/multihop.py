"""
MultihopSpatial (etri-vilab/MultihopSpatial) loader.

Thin wrapper around `datasets.load_dataset` — returns the raw HF Dataset.

Fields per sample (from the HF dataset card):
- image    : PIL.Image
- question : str
- answer   : str ('(c) frame of the reed picture'-style MCQ answer)
- bbox     : list[float], [x1, y1, x2, y2] in 0-100 scale (percent of image)
- hop      : str ('1hop' / '2hop' / '3hop')
"""
from __future__ import annotations

from pathlib import Path

REPO_ID = "etri-vilab/MultihopSpatial"


def load_multihop(
    split: str = "test",
    cache_dir: str | Path | None = None,
):
    """
    Parameters
    ----------
    split : 'test' (4,500), 'train' (6,791), or 'all' (returns a DatasetDict).
    cache_dir : Optional path. Defaults to HF's default cache.
    """
    from datasets import load_dataset

    cache = str(cache_dir) if cache_dir else None
    ds = load_dataset(REPO_ID, cache_dir=cache)
    if split == "all":
        return ds
    return ds[split]


__all__ = ["load_multihop", "REPO_ID"]
