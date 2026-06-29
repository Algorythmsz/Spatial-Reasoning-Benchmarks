# Spatial-benchmarks

Thin loading helpers for the four spatial reasoning benchmarks and the SpatialCorpus training corpus used in POST CRISP evaluation.

The four benchmarks have genuinely different schemas (point-in-mask / bbox IoU / MRA / MCQ), so each is exposed in its native form.

### Evaluation benchmarks

| Benchmark | HF repo | Samples | Output type | Scoring | Viewer |
|---|---|---|---|---|---|
| [SpatialScore](#spatialscore) | `haoningwu/SpatialScore` | 5,025 | mixed (MCQ / judge / distance / open) | per-qtype average | [SpatialScore_Viewer](https://algorythmsz.github.io/SpatialScore_Viewer/) |
| [MultihopSpatial](#multihopspatial) | `etri-vilab/MultihopSpatial` | 4,500 + 6,791 | MCQ + bbox | Acc@50IoU | [MultihopSpatial_Viewer](https://algorythmsz.github.io/MultihopSpatial_Viewer/) |
| [RefSpatial-Expand-Bench](#refspatial-expand-bench) | `JingkunAn/RefSpatial-Expand-Bench` | 441 (loc / place configs) | point | point-in-mask | [RefSpatial-Expand-bench_Viewer](https://algorythmsz.github.io/RefSpatial-Expand-bench_Viewer/) |
| [RefSpatial-Bench](#refspatial-bench) *(subset of Expand)* | `BAAI/RefSpatial-Bench` | 277 (loc / place / **unseen**) | point | point-in-mask | — |

### Training corpus

| Corpus | HF repo | Samples | Format | Size | Viewer |
|---|---|---|---|---|---|
| [SpatialCorpus](#spatialcorpus) | `haoningwu/SpatialCorpus` | ~331K | multimodal QA (SFT) | ~73.6 GB | [SpatialCorpus_Viewer](https://algorythmsz.github.io/SpatialCorpus_Viewer/) |

## Install

```bash
git clone https://github.com/Algorythmsz/Spatial-Reasoning-Benchmarks
cd spatial-benchmarks
pip install -e .
```

Requires Python ≥ 3.10. `huggingface_hub`, `datasets`, `pyarrow`, and `Pillow` are installed automatically.

---

## SpatialScore

**Repo**: [`haoningwu/SpatialScore`](https://huggingface.co/datasets/haoningwu/SpatialScore) — CVPR 2026 Highlight. arXiv:2505.17012. 

**Viewer**: [SpatialScore_Viewer](https://algorythmsz.github.io/SpatialScore_Viewer/)

5,025 manually-verified samples, organized into **30 sub_tasks** across 10 categories. Samples are aggregated from many source datasets (recorded in the `dataset` field — important for leakage analysis). The image archive is ~15.8 GB.

### Load

```python
from spatial_benchmarks import load_spatialscore

records = load_spatialscore()       # list[dict], 5,025 records
print(len(records))
print(records[0])
```

The first call also downloads `SpatialScore_benchmark.zip` (~15.8 GB) and extracts it — this takes a while, including on fast connections. Subsequent calls hit the cache.

To inspect the schema without downloading the images:

```python
records = load_spatialscore(extract_images=False, image_zip=None)
```

### Record schema

Each record is the raw HF dict, plus an `abs_image` key (absolute path after extraction) injected by the loader. Main fields:

| Field | Type | Description |
|---|---|---|
| `id` | int | sample id |
| `question` | str | question text |
| `answer` | str | gold answer (form depends on qtype) |
| `question_type` | str | `multi-choice` / `judgement` / `open-ended-distance` / `open-ended-general` |
| `options` | list\[str\] / None | choices when qtype is `multi-choice` |
| `sub_task` | str | one of 30 sub_tasks |
| `category` | str | one of 10 categories |
| `dataset` | str | source dataset (CVBench / spatialsense / SpatialScore-Repurpose / ...) — **critical for leakage analysis** |
| `image` | str / list\[str\] | relative path inside the zip |
| `abs_image` | Path / list\[Path\] | absolute path after extraction (injected by the loader) |
| `extra_info.answer_value` | float | numeric value for `open-ended-distance` |
| `extra_info.answer_unit` | str | unit (`m` / `cm` / `inch` / ...) |

### Categories & filtering

The 5,025 samples are organized into **10 categories** (30 sub_tasks total). Counts (verified against v1):

| Category | Count |
|---|---|
| Camera | 778 |
| Object Localization | 697 |
| Object Distance | 576 |
| Object Size | 559 |
| Depth Estimation | 520 |
| Mental Animation | 447 |
| View Reasoning | 446 |
| Object Motion | 415 |
| Counting | 315 |
| Temporal Reasoning | 272 |

The 30 sub_tasks span these categories. Counts (sorted by size):

| Sub_task | Count | | Sub_task | Count |
|---|---|---|---|---|
| Absolute Depth | 325 | | Camera Motion | 174 |
| Absolute Distance | 313 | | Appearance Order | 167 |
| Absolute Size | 281 | | Object Counting | 154 |
| Relative Distance | 263 | | Maze Navigation | 115 |
| Camera Extrinsics | 262 | | Object Motion | 113 |
| Homography Matrix | 246 | | Video Counting | 111 |
| View Perspective | 235 | | Navigation Route | 105 |
| Point Tracking | 229 | | Camera Intrinsics | 96 |
| Object Existence | 220 | | Size Compatibility | 89 |
| Spatial Map | 220 | | Velocity Estimation | 73 |
| Spatial Position | 215 | | 2D Localization | 50 |
| 3D Object Detection | 212 | | Count with Relation | 50 |
| Orientation | 211 | | Multi-view Projection | 46 |
| Relative Depth | 195 | | 2D/3D Rotation | 46 |
| Relative Size | 189 | | Space Folding | 20 |

To verify counts against your local copy, or to see the category → sub_task mapping:

```python
from collections import Counter
from itertools import groupby
from spatial_benchmarks import load_spatialscore

records = load_spatialscore(extract_images=False, image_zip=None)

# Category-level counts
print(Counter(r["category"] for r in records))

# Sub_task breakdown grouped by category
recs = sorted(records, key=lambda r: (r["category"], r["sub_task"]))
for cat, group in groupby(recs, key=lambda r: r["category"]):
    subs = Counter(r["sub_task"] for r in group)
    print(f"\n{cat}")
    for sub, n in subs.most_common():
        print(f"  {sub:30s} {n}")
```

Filter by category or sub_task:

```python
records = load_spatialscore()  # full load with images

distance       = [r for r in records if r["category"] == "Object Distance"]
absolute_depth = [r for r in records if r["sub_task"] == "Absolute Depth"]
```

### Scoring

The official eval (`test_qwen.py` in `haoningwu3639/SpatialScore`) uses four scorers, each producing a per-sample value in \[0, 1\]:

- **multi-choice**: exact-match binary (random = 25%)
- **judgement**: yes / no binary (random = 50%)
- **open-ended-distance**: MRA (Mean Relative Accuracy) — predicted distance vs ground-truth distance (random ≈ 0%)
- **open-ended-general**: LLM judge or rule-based string match

⚠️ **Caveat**: the three random baselines (25 / 50 / 0%) are very different, but per-sub_task scores are reported as a simple average across qtypes. When the qtype mix within a sub_task is skewed, the average is misleading. For POST CRISP analysis, always report a per-qtype breakdown alongside the sub_task mean.

### POST CRISP mapping

- **High-value sub_tasks**: Object Localization, 3D Positional Relation, Depth & Distance (good atomic-capability diagnostics)
- **Skip**: camera pose / homography / motion estimation (outside POST CRISP scope)
- **Leakage warning**: Object Counting is 51% CVBench, Spatial Position is 58% spatialsense — if any of these sources overlap with training data, the score is inflated. Use `record["dataset"]` to slice scores by source.
- **Leakage-safe primary metric**: `SpatialScore-Repurpose` (1,091 samples) — auto-generated from category labels, lower leakage risk.

---

## MultihopSpatial

**Repo**: [`etri-vilab/MultihopSpatial`](https://huggingface.co/datasets/etri-vilab/MultihopSpatial) — ETRI / KAIST 2026. arXiv:2603.18892. 

**Viewer**: [MultihopSpatial_Viewer](https://algorythmsz.github.io/MultihopSpatial_Viewer/)

4,500 eval + 6,791 train. 1 / 2 / 3-hop sequential spatial reasoning. All 4,500 eval samples are annotated by 10 trained human experts (Krippendorff α = 0.90).

### Load

```python
from spatial_benchmarks import load_multihop

test_ds  = load_multihop("test")     # 4,500
train_ds = load_multihop("train")    # 6,791
both     = load_multihop("all")      # DatasetDict {test, train}

print(test_ds)
print(test_ds[0])
```

Returns a `datasets.Dataset`, so `.filter()`, `.map()`, `.shuffle()` all work.

### Sample schema

| Field | Type | Description |
|---|---|---|
| `image` | PIL.Image | auto-decoded |
| `question` | str | e.g. "From the perspective of the woman holding the remote control, ..." |
| `answer` | str | "(c) frame of the reed picture" — MCQ answer as free text |
| `bbox` | list\[float\] | **\[x1, y1, x2, y2\] in 0–100 scale** (percent of image) |
| `hop` | str | "1hop" / "2hop" / "3hop" |

### Scoring

The paper's primary metric is **Acc@50IoU**: a sample counts as correct only if (a) the MCQ answer matches *and* (b) the predicted bbox has IoU ≥ 0.5 with the ground-truth bbox. This catches "lucky guesses" where the answer is right but the grounding is wrong.

Reference scoring snippet:

```python
def acc_at_iou50(samples, predictions):
    """predictions: list of dict with 'answer' (str) and 'bbox' ([x1, y1, x2, y2] in 0-100)."""
    correct = 0
    for s, p in zip(samples, predictions):
        if s["answer"].strip() != p["answer"].strip():
            continue
        if iou(s["bbox"], p["bbox"]) >= 0.5:
            correct += 1
    return correct / len(samples)
```

### POST CRISP mapping

- The bbox IoU signal mirrors Track A's (CA-1M extension) dense reward — same scoring shape, different supervision regime
- Per-hop accuracy directly measures the multi-hop training effect. Specifically: does the model that beats RoboRefer at 1-hop also beat it at 2 / 3-hop, or does the gap shrink?
- **Watch the bbox scale**: 0–100 (percent), *not* 0–1 and *not* pixels. Easy to get wrong when normalizing model outputs

---

## RefSpatial-Expand-Bench

**Repo**: [`JingkunAn/RefSpatial-Expand-Bench`](https://huggingface.co/datasets/JingkunAn/RefSpatial-Expand-Bench) 

**Viewer**: [RefSpatial-Expand-bench_Viewer](https://algorythmsz.github.io/RefSpatial-Expand-bench_Viewer/)

441 samples across multiple sources: 2D web + CA-1M + Sim (Infinigen + Objaverse-LVIS). **RefSpatial-Bench is a complete subset of this dataset** — all 277 original samples (location, placement, unseen) are included here.

| Config | Samples |
|---|---|
| location | 241 |
| placement | 200 |

Note: on the Hub these are *configs* (the `name=` argument to `load_dataset`), not splits.

### Load

```python
from spatial_benchmarks import load_refspatial_expand

loc_ds = load_refspatial_expand("location")   # 241
plc_ds = load_refspatial_expand("placement")  # 200
all_ds = load_refspatial_expand("all")        # 441, with an added 'config' column

print(all_ds[0])
```

### Sample schema

| Field | Type | Description |
|---|---|---|
| `id` | str | sample id |
| `rgb` | PIL.Image | RGB image |
| `mask` | PIL.Image | binary mask (same scoring as RefSpatial-Bench) |
| `object` | str | natural-language target description |
| `prompt` | str | full referring expression |
| `suffix` | str | answer-format instruction |
| `category` | str | source tag (Infinigen / Objaverse-LVIS / CA-1M / 2D-web / ...) |
| `step` | int | reasoning hops |
| `scene` | str | scene id (for sim data) |

### Scoring

**point-in-mask**: the model receives `prompt + suffix` and outputs normalized coordinates `[(x, y)]` with `x, y ∈ [0, 1]`. The prediction counts as correct if the point lies inside the 1-region of the mask.

```python
import numpy as np

def point_in_mask(pred_point, mask_pil):
    """pred_point: (x, y) in [0, 1]. mask_pil: PIL.Image binary mask."""
    mask = np.array(mask_pil.convert("L")) > 128
    h, w = mask.shape
    px, py = int(pred_point[0] * w), int(pred_point[1] * h)
    if not (0 <= px < w and 0 <= py < h):
        return False
    return bool(mask[py, px])

# Eval loop
correct = sum(
    point_in_mask(pred[i], all_ds[i]["mask"])
    for i in range(len(all_ds))
)
acc = correct / len(all_ds)
```

### POST CRISP mapping

- **Primary point-in-mask benchmark**: broader coverage than RefSpatial-Bench (441 vs 277 samples, multiple real + sim sources)
- Slicing scores by `category` reveals the sim-to-real transfer gap between Infinigen/Objaverse-LVIS and CA-1M/2D-web sources
- Per-step accuracy measures compositional generalization: step = 1 (atomic) vs step = 2, 3 (composed)

---

## RefSpatial-Bench

**Repo**: [`BAAI/RefSpatial-Bench`](https://huggingface.co/datasets/BAAI/RefSpatial-Bench) — arXiv:2506.04308 (RoboRefer).

> **RefSpatial-Bench is a complete subset of RefSpatial-Expand-Bench.** The viewer is maintained only for Expand: [algorythmsz.github.io/RefSpatial-Expand-bench_Viewer](https://algorythmsz.github.io/RefSpatial-Expand-bench_Viewer/)

277 samples, 3 splits. Contains an **unseen** split (77 samples) of spatial-relation combinations RoboRefer never saw — the strongest OOD generalization signal.

| Split | Samples | Purpose |
|---|---|---|
| location | 100 | inside RoboRefer's training distribution (location relations) |
| placement | 100 | inside RoboRefer's training distribution (placement relations) |
| **unseen** | **77** | spatial-relation combinations RoboRefer never saw — **OOD generalization** |

### Load

```python
from spatial_benchmarks import load_refspatial_bench

unseen_ds = load_refspatial_bench("unseen")     # 77 samples ← OOD split
all_ds    = load_refspatial_bench("all")        # DatasetDict {location, placement, unseen}

print(unseen_ds[0])
```

### Sample schema

| Field | Type | Description |
|---|---|---|
| `id` | int | 0–99 (resets per split) |
| `image` | PIL.Image | 384–640 px |
| `mask` | PIL.Image | binary mask of the correct region (used for scoring) |
| `object` | str | natural-language target description ("the orange box") |
| `prompt` | str | full referring expression ("Please point out the orange box.") |
| `suffix` | str | answer-format instruction; one constant value across all samples |
| `step` | int | 1 / 2 / 3 — reasoning hops |

### Scoring

Identical to RefSpatial-Expand-Bench (point-in-mask). The snippet above applies as-is.

### POST CRISP mapping

- The **77 samples in the unseen split** are the paper's OOD evaluation: relation combinations disjoint from RoboRefer's training distribution
- POST CRISP deliberately does **not** train on RefSpatial-20M, both for fair head-to-head comparison with RoboRefer and to keep the contribution story clean ("if you train on their data, you can't separate the method effect from the data effect")
- Per-step accuracy is a direct measure of compositional generalization: step = 1 (atomic) vs step = 2, 3 (composed). If POST CRISP degrades less than RoboRefer as `step` increases, the filter-chain structure is doing work
- ⚠️ The dataset card warns: "If your model is not trained with RefSpatial, the Unseen set should not be used for evaluation." The authors meant unseen-relative-to-RefSpatial-training. Using it as general OOD evaluation is defensible — but the paper must explicitly state that interpretation

---

## SpatialCorpus

**Repo**: [`haoningwu/SpatialCorpus`](https://huggingface.co/datasets/haoningwu/SpatialCorpus) — CVPR 2026 Highlight. arXiv:2505.17012. 

**Viewer**: [SpatialCorpus_Viewer](https://algorythmsz.github.io/SpatialCorpus_Viewer/)

~331K multimodal QA samples for supervised fine-tuning of vision-language models on spatial reasoning tasks. Covers the same 10 categories and 30 sub_tasks as SpatialScore. The image archive is ~73.6 GB.

⚠️ **This is training data, not an evaluation benchmark.** For POST CRISP, its primary uses are leakage analysis (checking if benchmark samples appear in the corpus) and training-set ablations (sub_task coverage).

### Load

```python
from spatial_benchmarks import load_spatialcorpus

# Annotations only (zip downloaded, images not extracted — faster startup)
records = load_spatialcorpus(extract_images=False)
print(len(records))   # ~331,000
print(records[0].keys())

# Full load with images (~73.6 GB extraction)
records = load_spatialcorpus()
```

The annotation is bundled inside `SpatialCorpus.zip`. On first call the full archive is downloaded regardless of `extract_images` — the zip is the only file in the repo. Subsequent calls use the cache.

### Record schema

SpatialCorpus records share the core fields with SpatialScore. Exact field names are confirmed at extraction time — use `records[0].keys()` to inspect after loading. Expected fields:

| Field | Type | Description |
|---|---|---|
| `id` | int / str | sample id |
| `question` | str | question text |
| `answer` | str | gold answer |
| `question_type` | str | `multi-choice` / `judgement` / `open-ended-distance` / `open-ended-general` |
| `options` | list\[str\] / None | choices when qtype is `multi-choice` |
| `sub_task` | str | one of 30 sub_tasks |
| `category` | str | one of 10 categories |
| `dataset` | str | source dataset (important for leakage analysis) |
| `image` | str / list\[str\] | relative path inside the zip |
| `abs_image` | Path / list\[Path\] | absolute path after extraction (injected by loader) |

SFT-specific fields (e.g. `conversations`, `system_prompt`) may also be present depending on the version of the corpus.

### POST CRISP mapping

- **Leakage analysis**: compare `record["dataset"]` values in the corpus against the same field in SpatialScore to quantify overlap. The `id` field is a secondary check for exact-sample overlap.
- **Training coverage**: count corpus samples per `sub_task` and compare to benchmark coverage. Sub_tasks that are underrepresented in the corpus but prominent in the benchmark are natural ablation axes.
- **POST CRISP does not fine-tune on SpatialCorpus** — the contribution story is about the filter-chain and data curation method, not data scale. SpatialCorpus is useful here only as a reference for what the original SpatialScore authors consider "good training data."

---

## POST CRISP — which metric goes where

The four benchmarks line up as an ablation gradient from atomic to compositional: **SpatialScore** diagnoses atomic capability (30 sub_tasks, per-qtype breakdown, source-dataset slicing for leakage), **MultihopSpatial** tests bbox-grounded multi-hop reasoning (Acc@50IoU per hop), **RefSpatial-Bench unseen** measures compositional OOD generalization (point-in-mask, step 1 → 3 degradation vs RoboRefer), and **RefSpatial-Expand-Bench** serves as a sim-to-real auxiliary (slice by `category`). POST CRISP never trains on RefSpatial-Bench or RefSpatial-20M, which keeps the head-to-head comparison with RoboRefer fair and the contribution story clean.

## License

Code: Apache-2.0. Each benchmark's data follows its original license.

## References

- SpatialScore: Wu et al., arXiv:2505.17012
- MultihopSpatial: Lee et al., arXiv:2603.18892
- RefSpatial-Bench / RoboRefer: arXiv:2506.04308
- CRISP (parent framework): anonymized repo `anon-share-legobench-E01A`
