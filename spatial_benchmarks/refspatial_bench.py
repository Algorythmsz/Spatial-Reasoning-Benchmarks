"""
RefSpatial-Bench (BAAI/RefSpatial-Bench) loader.

Thin wrapper around `datasets.load_dataset` — returns the raw HF Dataset.

Splits:
- location  (100)
- placement (100)
- unseen    ( 77)  ← strongest OOD evidence point for POST CRISP

Fields per sample (from the HF dataset viewer):
- id      : int (0-99 within each split)
- image   : PIL.Image (384-640 px)
- mask    : PIL.Image (binary mask of the correct region)
- object  : str (target description extracted from the prompt)
- prompt  : str (full referring expression)
- suffix  : str (answer-format instruction; one constant value across all samples)
- step    : int (1-3, reasoning hops)

Scoring: point-in-mask. The model outputs [(x, y)] in [0, 1]; correct if the
point lies inside the 1-region of the mask.
"""
from __future__ import annotations

from pathlib import Path

REPO_ID = "BAAI/RefSpatial-Bench"
SPLITS = ("location", "placement", "unseen")


def load_refspatial_bench(
    split: str = "all",
    cache_dir: str | Path | None = None,
):
    """
    Parameters
    ----------
    split : 'location' / 'placement' / 'unseen' / 'all' (returns a DatasetDict).
    cache_dir : Optional path. Defaults to HF's default cache.
    """
    from datasets import load_dataset

    assert split == "all" or split in SPLITS, (
        f"unknown split: {split!r}, choose from {SPLITS + ('all',)}"
    )
    cache = str(cache_dir) if cache_dir else None
    ds = load_dataset(REPO_ID, cache_dir=cache)
    if split == "all":
        return ds
    return ds[split]


__all__ = ["load_refspatial_bench", "REPO_ID", "SPLITS"]
