"""
spatial_benchmarks
==================

Loading helpers for the four spatial reasoning benchmarks and the SpatialCorpus
training corpus used in POST CRISP.

Each benchmark is exposed with its native schema (no unified wrapper). See
README.md for the field-by-field documentation.

>>> from spatial_benchmarks import (
...     load_spatialscore,
...     load_multihop,
...     load_refspatial_expand,
...     load_refspatial_bench,
...     load_spatialcorpus,
... )
"""
from .multihop import load_multihop
from .refspatial_bench import load_refspatial_bench
from .refspatial_expand import load_refspatial_expand
from .spatialcorpus import load_spatialcorpus
from .spatialscore import load_spatialscore

__version__ = "0.3.0"

__all__ = [
    "load_spatialscore",
    "load_multihop",
    "load_refspatial_expand",
    "load_refspatial_bench",
    "load_spatialcorpus",
    "__version__",
]
