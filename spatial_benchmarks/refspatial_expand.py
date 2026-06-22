"""
RefSpatial-Expand-Bench (JingkunAn/RefSpatial-Expand-Bench) loader.

NOTE: on the Hub, 'location' / 'placement' are *configs* (the `name=` argument
to load_dataset), not splits. Inside each config the actual split is 'train'.
To keep the user-facing API simple we treat configs the same way the user
would expect splits to work.

Fields per sample:
- id        : str
- object    : str
- prompt    : str
- suffix    : str
- rgb       : PIL.Image
- mask      : PIL.Image
- category  : str (Infinigen / Objaverse-LVIS / CA-1M / 2D-web, ...)
- step      : int (reasoning hops)
- scene     : str (scene id when the sample comes from a simulator)

Sources: 2D web + CA-1M + Sim (Infinigen + Objaverse-LVIS).

Scoring: same as RefSpatial-Bench (point-in-mask).
"""
from __future__ import annotations

from pathlib import Path

REPO_ID = "JingkunAn/RefSpatial-Expand-Bench"
CONFIGS = ("location", "placement")


def load_refspatial_expand(
    config: str = "location",
    cache_dir: str | Path | None = None,
):
    """
    Parameters
    ----------
    config : 'location' (241), 'placement' (200), or 'all'.
        When 'all', the two configs are concatenated into a single Dataset and a
        'config' column is added to each row recording which one it came from.
    cache_dir : Optional path. Defaults to HF's default cache.
    """
    from datasets import concatenate_datasets, load_dataset

    assert config == "all" or config in CONFIGS, (
        f"unknown config: {config!r}, choose from {CONFIGS + ('all',)}"
    )
    cache = str(cache_dir) if cache_dir else None

    if config != "all":
        d = load_dataset(REPO_ID, name=config, cache_dir=cache)
        inner_split = "train" if "train" in d else list(d.keys())[0]
        return d[inner_split]

    # 'all' = concat both configs, tag each row with its origin
    parts = []
    for cfg in CONFIGS:
        d = load_dataset(REPO_ID, name=cfg, cache_dir=cache)
        inner_split = "train" if "train" in d else list(d.keys())[0]
        ds = d[inner_split].add_column("config", [cfg] * len(d[inner_split]))
        parts.append(ds)
    return concatenate_datasets(parts)


__all__ = ["load_refspatial_expand", "REPO_ID", "CONFIGS"]
