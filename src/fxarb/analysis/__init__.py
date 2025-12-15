"""Statistical analysis tools for cointegration and stationarity."""

from .cointegration import (
    johansen_test,
    construct_spread,
    estimate_half_life,
    CointegrationResult,
)
from .stationarity import adf_test, kpss_test, StationarityResult
from .triplets import (
    enumerate_triplets,
    validate_triplet,
    score_triplet,
    find_best_triplets,
    TripletScore,
)

__all__ = [
    "johansen_test",
    "construct_spread",
    "estimate_half_life",
    "CointegrationResult",
    "adf_test",
    "kpss_test",
    "StationarityResult",
    "enumerate_triplets",
    "validate_triplet",
    "score_triplet",
    "find_best_triplets",
    "TripletScore",
]
