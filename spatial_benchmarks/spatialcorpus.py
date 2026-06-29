"""
SpatialCorpus (haoningwu/SpatialCorpus) loader.

SpatialCorpus is the SFT training corpus that accompanies the SpatialScore
benchmark (arXiv:2505.17012, CVPR 2026 Highlight). It contains ~331K
multimodal QA samples covering the same 10 categories and 30 sub_tasks as
SpatialScore, formatted for supervised fine-tuning of vision-language models.

Unlike the four evaluation benchmarks, this is *training data*. The primary
use cases in POST CRISP are:
  - leakage analysis (do benchmark samples appear in training data?)
  - training-set ablations (which sub_tasks does the corpus cover?)

Important: the annotation file is bundled inside SpatialCorpus.zip. The full
~73.6 GB archive is downloaded on first use. Subsequent calls hit the cache.

    records = load_spatialcorpus()
    # records: list[dict] — annotation records + abs_image key
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "haoningwu/SpatialCorpus"

# Annotation filename candidates (tried in order, both as standalone HF files
# and as paths inside SpatialCorpus.zip).
_ANNOT_CANDIDATES = [
    "SpatialCorpus.ndjson",
    "SpatialCorpus.json",
    "SpatialCorpus_train.ndjson",
    "SpatialCorpus_train.json",
]


def load_spatialcorpus(
    cache_dir: str | Path | None = None,
    annotation_file: str | None = None,
    image_zip: str | None = "SpatialCorpus.zip",
    extract_images: bool = True,
) -> list[dict[str, Any]]:
    """
    Download SpatialCorpus from the Hub and return the records as a list of dicts.

    SpatialCorpus is the SFT training corpus for the SpatialScore benchmark
    (~331K multimodal QA samples, same task structure as SpatialScore).

    The annotation is bundled inside the zip. The loader first checks whether
    a standalone annotation file exists in the HF repo (fast path); if not, it
    downloads the full archive and extracts just the annotation file.

    Parameters
    ----------
    cache_dir : Download cache. Defaults to ~/.cache/spatial_benchmarks.
    annotation_file : Explicit annotation filename (path inside the repo or
        inside the zip). If None the loader auto-detects from _ANNOT_CANDIDATES.
    image_zip : Archive filename on the Hub (``SpatialCorpus.zip``, ~73.6 GB).
        Set to None to skip download — only works if the zip is already cached
        and the annotation has already been extracted to cache_dir.
    extract_images : If True, extract the full archive and populate ``abs_image``
        on each record. If False, only the annotation file is extracted (fast),
        and ``abs_image`` is not added. The full archive is still downloaded
        either way if the annotation lives inside the zip.

    Notes
    -----
    ⚠️  First call downloads ~73.6 GB (the full archive, since annotations are
    bundled inside it). Use ``extract_images=False`` to skip image extraction
    and get annotations faster once the archive is cached.
    """
    cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "spatial_benchmarks"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1) Annotation file — standalone HF file if available, else from zip
    ann_path = _get_annotation(annotation_file, image_zip, cache_dir)

    # 2) Optionally extract full image archive
    image_root: Path | None = None
    if image_zip and extract_images:
        zip_path = Path(hf_hub_download(
            repo_id=REPO_ID,
            filename=image_zip,
            repo_type="dataset",
            cache_dir=str(cache_dir),
        ))
        image_root = cache_dir / "spatialcorpus_images" / Path(image_zip).stem
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

def _get_annotation(
    annotation_file: str | None,
    image_zip: str | None,
    cache_dir: Path,
) -> Path:
    """Return a local path to the annotation file, downloading if needed."""
    # Explicit override — download directly from HF
    if annotation_file is not None:
        return Path(hf_hub_download(
            repo_id=REPO_ID,
            filename=annotation_file,
            repo_type="dataset",
            cache_dir=str(cache_dir),
        ))

    # Check for a standalone annotation in the HF repo (fast API call)
    try:
        repo_files = set(HfApi().list_repo_files(repo_id=REPO_ID, repo_type="dataset"))
    except Exception:
        repo_files = set()

    for candidate in _ANNOT_CANDIDATES:
        if candidate in repo_files:
            return Path(hf_hub_download(
                repo_id=REPO_ID,
                filename=candidate,
                repo_type="dataset",
                cache_dir=str(cache_dir),
            ))

    # Fallback: extract annotation from inside the zip
    if image_zip is None:
        raise FileNotFoundError(
            f"No standalone annotation found in {REPO_ID} and image_zip=None. "
            "Set image_zip='SpatialCorpus.zip' so the archive can be downloaded."
        )
    zip_path = Path(hf_hub_download(
        repo_id=REPO_ID,
        filename=image_zip,
        repo_type="dataset",
        cache_dir=str(cache_dir),
    ))
    return _extract_annotation_from_zip(zip_path, cache_dir)


def _extract_annotation_from_zip(zip_path: Path, cache_dir: Path) -> Path:
    """Extract only the annotation file from the zip (fast partial extraction)."""
    ann_dir = cache_dir / "spatialcorpus_ann"
    ann_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        target: str | None = None

        for candidate in _ANNOT_CANDIDATES:
            for name in names:
                if Path(name).name == candidate:
                    target = name
                    break
            if target:
                break

        # Fallback: any shallow .ndjson / .json
        if target is None:
            for name in names:
                p = Path(name)
                if p.suffix in (".ndjson", ".json") and len(p.parts) <= 2:
                    target = name
                    break

        if target is None:
            shallow = [n for n in names[:40] if n.count("/") <= 1]
            raise FileNotFoundError(
                f"Could not find annotation file in {zip_path}. "
                f"Checked candidates: {_ANNOT_CANDIDATES}. "
                f"Zip top-level entries: {shallow}"
            )

        out_path = ann_dir / Path(target).name
        if not out_path.exists():
            with zf.open(target) as src, open(out_path, "wb") as dst:
                dst.write(src.read())

    return out_path


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
    (e.g. 'SpatialCorpus/foo.jpg' vs 'foo.jpg'). Try both.
    """
    p1 = image_root / rel
    if p1.exists():
        return p1
    stripped = rel.split("/", 1)[-1] if "/" in rel else rel
    p2 = image_root / stripped
    if p2.exists():
        return p2
    return p1


__all__ = ["load_spatialcorpus", "REPO_ID"]
