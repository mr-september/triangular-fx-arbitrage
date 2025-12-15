"""
Triplet discovery and validation for triangular FX arbitrage.

Enumerates valid currency triplets and scores them by cointegration strength.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import pandas as pd

from .cointegration import CointegrationResult, construct_spread, estimate_half_life, johansen_test
from .stationarity import adf_test


# Standard currency pairs and their components
MAJOR_CURRENCIES = ["EUR", "USD", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD"]

# Mapping of pair codes to (base, quote) currencies
def parse_pair(pair: str) -> tuple[str, str]:
    """Parse 6-letter pair code into (base, quote) currencies."""
    pair = pair.upper()
    return pair[:3], pair[3:]


def pair_code(base: str, quote: str) -> str:
    """Create pair code from base and quote currencies."""
    return f"{base.lower()}{quote.lower()}"


@dataclass
class TripletScore:
    """Scoring results for a currency triplet."""

    pair1: str
    pair2: str
    pair3: str
    cointegration_result: CointegrationResult
    spread_adf_pvalue: float
    half_life: float | None
    score: float  # Overall score (higher is better)

    def __str__(self) -> str:
        return (
            f"Triplet: {self.pair1}/{self.pair2}/{self.pair3}\n"
            f"  Cointegrating relationships: {self.cointegration_result.n_cointegrating}\n"
            f"  Spread ADF p-value: {self.spread_adf_pvalue:.6f}\n"
            f"  Half-life: {self.half_life:.1f} bars\n"
            f"  Score: {self.score:.4f}"
        )

    @property
    def is_cointegrated(self) -> bool:
        """Whether the triplet has at least one cointegrating relationship."""
        return self.cointegration_result.n_cointegrating > 0


def enumerate_triplets(available_pairs: list[str]) -> list[tuple[str, str, str]]:
    """
    Enumerate all valid FX triplets from available pairs.

    A valid triplet (A/C, A/B, B/C) satisfies the triangular relationship:
    A/C = A/B * B/C (in log space: log(A/C) = log(A/B) + log(B/C))

    Args:
        available_pairs: List of available pair codes (e.g., ['eurusd', 'gbpusd'])

    Returns:
        List of valid triplet tuples, each containing 3 pair codes
    """
    # Parse all pairs into currency components
    pairs_parsed = {}
    for pair in available_pairs:
        pair = pair.lower()
        base, quote = parse_pair(pair)
        pairs_parsed[pair] = (base, quote)

    # Build currency graph: which currencies connect to which
    currency_pairs: dict[str, set[str]] = {}
    for pair, (base, quote) in pairs_parsed.items():
        if base not in currency_pairs:
            currency_pairs[base] = set()
        if quote not in currency_pairs:
            currency_pairs[quote] = set()
        currency_pairs[base].add(quote)
        currency_pairs[quote].add(base)

    # Find all triplets
    triplets = []

    # For each pair of currencies (A, C), find common connections (B)
    currencies = list(currency_pairs.keys())
    for a, c in combinations(currencies, 2):
        # Find currencies B that connect to both A and C
        common = currency_pairs[a] & currency_pairs[c]

        for b in common:
            if b == a or b == c:
                continue

            # Check if we have all three pairs
            # Need: A/C (or C/A), A/B (or B/A), B/C (or C/B)
            def find_pair(base: str, quote: str) -> str | None:
                p1 = pair_code(base, quote)
                p2 = pair_code(quote, base)
                if p1 in pairs_parsed:
                    return p1
                if p2 in pairs_parsed:
                    return p2
                return None

            pair_ac = find_pair(a, c)
            pair_ab = find_pair(a, b)
            pair_bc = find_pair(b, c)

            if all([pair_ac, pair_ab, pair_bc]):
                triplet = tuple(sorted([pair_ac, pair_ab, pair_bc]))
                if triplet not in triplets:
                    triplets.append(triplet)

    return [tuple(t) for t in triplets]


def validate_triplet(pair1: str, pair2: str, pair3: str) -> bool:
    """
    Validate that three pairs form a valid triangular relationship.

    Args:
        pair1, pair2, pair3: Currency pair codes

    Returns:
        True if the triplet is valid
    """
    currencies = set()
    for pair in [pair1, pair2, pair3]:
        base, quote = parse_pair(pair)
        currencies.add(base)
        currencies.add(quote)

    # A valid triplet should involve exactly 3 currencies
    return len(currencies) == 3


def score_triplet(
    data: dict[str, pd.DataFrame],
    max_half_life: float = 500.0,
) -> TripletScore:
    """
    Score a triplet based on cointegration strength and spread properties.

    Scoring criteria:
    1. Number of cointegrating relationships (must be >= 1)
    2. ADF p-value of the spread (lower is better)
    3. Half-life of mean reversion (shorter is better, but too short may be noise)

    Args:
        data: Dict mapping pair codes to DataFrames with log_price or close columns
        max_half_life: Maximum acceptable half-life (used for normalization)

    Returns:
        TripletScore with detailed results
    """
    pairs = list(data.keys())
    if len(pairs) != 3:
        raise ValueError(f"Expected 3 pairs, got {len(pairs)}")

    # Run Johansen test
    coint_result = johansen_test(data)

    # If no cointegration, return low score
    if coint_result.n_cointegrating == 0:
        return TripletScore(
            pair1=pairs[0],
            pair2=pairs[1],
            pair3=pairs[2],
            cointegration_result=coint_result,
            spread_adf_pvalue=1.0,
            half_life=None,
            score=0.0,
        )

    # Construct spread and test stationarity
    spread = construct_spread(data, hedge_ratios=coint_result.hedge_ratios)
    adf_result = adf_test(spread)

    # Estimate half-life
    try:
        half_life = estimate_half_life(spread)
    except ValueError:
        half_life = max_half_life

    # Calculate score
    # Score components (all normalized to 0-1, higher is better):
    # 1. Cointegration bonus: 0.3 for 1 relationship, 0.5 for 2+
    coint_score = 0.3 if coint_result.n_cointegrating == 1 else 0.5

    # 2. ADF score: transform p-value (lower p-value -> higher score)
    adf_score = 1.0 - min(adf_result.pvalue, 1.0)

    # 3. Half-life score: prefer moderate half-life (10-100 bars)
    # Too short might be noise, too long is hard to trade
    if half_life is None:
        hl_score = 0.0
    elif half_life < 5:
        hl_score = half_life / 10  # Penalize very short
    elif half_life <= 100:
        hl_score = 1.0  # Sweet spot
    else:
        hl_score = max(0, 1.0 - (half_life - 100) / max_half_life)

    # Weighted combination
    score = 0.3 * coint_score + 0.4 * adf_score + 0.3 * hl_score

    return TripletScore(
        pair1=pairs[0],
        pair2=pairs[1],
        pair3=pairs[2],
        cointegration_result=coint_result,
        spread_adf_pvalue=adf_result.pvalue,
        half_life=half_life,
        score=score,
    )


def find_best_triplets(
    data_loader,
    available_pairs: list[str],
    top_n: int = 5,
    timeframe: str = "5min",
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[TripletScore]:
    """
    Find the best cointegrated triplets from available data.

    Args:
        data_loader: Function to load pair data (signature: loader(pair) -> DataFrame)
        available_pairs: List of available pair codes
        top_n: Number of top triplets to return
        timeframe: Timeframe for analysis
        start_date: Optional start date filter
        end_date: Optional end date filter

    Returns:
        List of TripletScore objects, sorted by score (descending)
    """
    triplets = enumerate_triplets(available_pairs)
    scores = []

    for pair1, pair2, pair3 in triplets:
        try:
            # Load data for each pair
            data = {
                pair1: data_loader(pair1),
                pair2: data_loader(pair2),
                pair3: data_loader(pair3),
            }

            # Score the triplet
            score = score_triplet(data)
            scores.append(score)

        except Exception as e:
            # Skip triplets with data issues
            print(f"Skipping {pair1}/{pair2}/{pair3}: {e}")
            continue

    # Sort by score (descending) and return top N
    scores.sort(key=lambda x: x.score, reverse=True)

    return scores[:top_n]


# Pre-defined triplets known to work well
RECOMMENDED_TRIPLETS = [
    ("eurusd", "gbpusd", "eurgbp"),  # EUR-GBP-USD: Most liquid
    ("eurusd", "usdjpy", "eurjpy"),  # EUR-USD-JPY: Major crosses
    ("gbpusd", "usdjpy", "gbpjpy"),  # GBP-USD-JPY: Volatile crosses
    ("audusd", "nzdusd", "audnzd"),  # AUD-NZD-USD: Commodity currencies
    ("eurusd", "usdchf", "eurchf"),  # EUR-USD-CHF: Safe haven
]
