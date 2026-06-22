"""
SpatialScore (haoningwu/SpatialScore) loader.

Unlike the other three benchmarks, this one cannot be loaded with a single
`datasets.load_dataset()` call: annotations live in NDJSON / JSON and images
ship as a separate zip archive. Everything is wrapped behind one function:

    records = load_spatialscore()
    # records: list[dict] — raw HF records, plus an `abs_image` key
    # pointing at the extracted image path.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "haoningwu/SpatialScore"

# Candidate annotation filenames, tried in order if `annotation_file` is None.
# As of the v1 update (2026.5), only `SpatialScore_benchmark.ndjson` exists at
# the repo root. Older v0 files (SpatialScore.json, SpatialScore-Hard.json,
# VGBench.json) are no longer published — VGBench and Hard were folded into
# the unified benchmark.
_ANNOT_CANDIDATES = [
    "SpatialScore_benchmark.ndjson",
]


def load_spatialscore(
    cache_dir: str | Path | None = None,
    annotation_file: str | None = None,
    image_zip: str | None = "SpatialScore_benchmark.zip",
    extract_images: bool = True,
) -> list[dict[str, Any]]:
    """
    Download SpatialScore from the Hub and return the records as a list of dicts.

    Each record is the raw HF dict with an extra ``abs_image`` key:
    - if ``record["image"]`` is a str, ``abs_image`` is a Path
    - if it is a list of strs, ``abs_image`` is a list of Paths
    - if ``extract_images=False``, ``abs_image`` is not added

    Parameters
    ----------
    cache_dir : Download cache. Defaults to ~/.cache/spatial_benchmarks.
    annotation_file : Filename of the NDJSON / JSON record file. If None, the
        first one that exists in the repo is picked. As of v1 (2026.5) only
        ``SpatialScore_benchmark.ndjson`` is published.
    image_zip : Filename of the image archive. ``SpatialScore_benchmark.zip``
        is the current v1 default (~15.8 GB, covering the 5,025 records). Set
        to None to skip the image archive entirely (annotations only).
    extract_images : If True, unzip the archive and populate ``abs_image``.
    """
    cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "spatial_benchmarks"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1) Annotation file
    if annotation_file is None:
        annotation_file = _auto_pick_annotation()
    ann_path = Path(hf_hub_download(
        repo_id=REPO_ID,
        filename=annotation_file,
        repo_type="dataset",
        cache_dir=str(cache_dir),
    ))

    # 2) Image archive
    image_root: Path | None = None
    if image_zip:
        zip_path = Path(hf_hub_download(
            repo_id=REPO_ID,
            filename=image_zip,
            repo_type="dataset",
            cache_dir=str(cache_dir),
        ))
        if extract_images:
            image_root = cache_dir / "spatialscore_images" / Path(image_zip).stem
            _ensure_unzipped(zip_path, image_root)

    # 3) Parse (NDJSON or JSON-array — auto-detected)
    records = _load_records(ann_path)

    # 4) Inject absolute image paths
    if image_root is not None:
        for rec in records:
            img_field = rec.get("image") or rec.get("images")
            if img_field is None:
                rec["abs_image"] = None
            elif isinstance(img_field, str):
                rec["abs_image"] = _resolve_image_path(image_root, img_field)
            elif isinstance(img_field, list):
                rec["abs_image"] = [_resolve_image_path(image_root, p) for p in img_field]

    return records


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _auto_pick_annotation() -> str:
    """Pick the first existing file from _ANNOT_CANDIDATES in the repo listing."""
    try:
        files = set(HfApi().list_repo_files(repo_id=REPO_ID, repo_type="dataset"))
    except Exception:
        return _ANNOT_CANDIDATES[0]
    for c in _ANNOT_CANDIDATES:
        if c in files:
            return c
    raise FileNotFoundError(
        f"None of {_ANNOT_CANDIDATES} found in {REPO_ID}. "
        f"Available files: {sorted(files)[:20]}..."
    )


def _load_records(path: Path) -> list[dict[str, Any]]:
    """Parse either NDJSON or JSON-array format."""
    text = path.read_text(encoding="utf-8")
    s = text.lstrip()
    if s.startswith("["):
        return json.loads(text)
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _ensure_unzipped(zip_path: Path, target_dir: Path) -> Path:
    """Unzip into target_dir. Skip if target_dir is non-empty."""
    target_dir.mkdir(parents=True, exist_ok=True)
    if any(target_dir.iterdir()):
        return target_dir
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)
    return target_dir


def _resolve_image_path(image_root: Path, rel: str) -> Path:
    """
    The path inside the zip may or may not include a top-level dir prefix
    (e.g. 'SpatialScore/foo.jpg' vs 'foo.jpg'). Try both.
    """
    p1 = image_root / rel
    if p1.exists():
        return p1
    stripped = rel.split("/", 1)[-1] if "/" in rel else rel
    p2 = image_root / stripped
    if p2.exists():
        return p2
    # Fall back to the first candidate so the caller can debug.
    return p1


__all__ = ["load_spatialscore", "REPO_ID"]
