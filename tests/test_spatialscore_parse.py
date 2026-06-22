"""Sanity checks for the SpatialScore NDJSON / JSON-array parsing helpers."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from spatial_benchmarks.spatialscore import (
    _load_records,
    _resolve_image_path,
)


def _make_records():
    return [
        {
            "id": 1,
            "question": "How far is the chair from the table?",
            "answer": "0.8 m",
            "question_type": "open-ended-distance",
            "sub_task": "Absolute Depth",
            "dataset": "CVBench",
            "image": "sub/img_001.jpg",
            "extra_info": {"answer_value": 0.8, "answer_unit": "m"},
        },
        {
            "id": 2,
            "question": "Is the cup on the left of the bottle?",
            "answer": "Yes",
            "question_type": "judgement",
            "sub_task": "Spatial Position",
            "dataset": "spatialsense",
            "image": ["sub/img_002.jpg"],
        },
    ]


def test_load_ndjson(tmp_path: Path):
    p = tmp_path / "x.ndjson"
    p.write_text("\n".join(json.dumps(r) for r in _make_records()))
    recs = _load_records(p)
    assert len(recs) == 2
    assert recs[0]["id"] == 1
    assert recs[1]["question_type"] == "judgement"


def test_load_json_array(tmp_path: Path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps(_make_records()))
    recs = _load_records(p)
    assert len(recs) == 2


def test_load_handles_blank_lines(tmp_path: Path):
    p = tmp_path / "x.ndjson"
    lines = [json.dumps(_make_records()[0]), "", "  ", json.dumps(_make_records()[1])]
    p.write_text("\n".join(lines))
    recs = _load_records(p)
    assert len(recs) == 2


def test_resolve_image_path_direct(tmp_path: Path):
    root = tmp_path / "img"
    (root / "sub").mkdir(parents=True)
    f = root / "sub" / "x.jpg"
    f.touch()
    p = _resolve_image_path(root, "sub/x.jpg")
    assert p == f


def test_resolve_image_path_strip_prefix(tmp_path: Path):
    """The path inside the zip may have a top-level dir prefix that doesn't exist on disk."""
    root = tmp_path / "img"
    root.mkdir()
    f = root / "x.jpg"
    f.touch()
    p = _resolve_image_path(root, "SpatialScore/x.jpg")
    assert p == f


def test_resolve_image_path_missing(tmp_path: Path):
    """If nothing is found, return the first candidate so the caller can debug."""
    root = tmp_path / "img"
    root.mkdir()
    p = _resolve_image_path(root, "nonexistent.jpg")
    assert p == root / "nonexistent.jpg"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
