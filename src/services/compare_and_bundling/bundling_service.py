from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Iterable, Tuple

import math
import pandas as pd
import numpy as np


@dataclass
class BundleStrategyConfig:
    name: str
    display_name: str
    w_value: float
    w_fit: float
    w_div: float
    w_risk: float


@dataclass
class BundleResult:
    titles: List[Dict[str, Any]]
    bundle_price_raw: float
    bundle_price_score: float
    bundle_fit_score: float
    bundle_diversity_score: float
    bundle_risk_score: float
    total_score: float
    strategy_name: str
    rationale: str


# ---------- Strategy config -------------------------------------------------


_STRATEGIES: Dict[str, BundleStrategyConfig] = {
    "max_value": BundleStrategyConfig(
        name="max_value",
        display_name="Maximize Value",
        w_value=0.6,
        w_fit=0.2,
        w_div=0.1,
        w_risk=0.1,
    ),
    "balanced": BundleStrategyConfig(
        name="balanced",
        display_name="Balanced Value & Risk",
        w_value=0.35,
        w_fit=0.25,
        w_div=0.2,
        w_risk=0.2,
    ),
    "min_risk": BundleStrategyConfig(
        name="min_risk",
        display_name="Minimize Risk",
        w_value=0.25,
        w_fit=0.2,
        w_div=0.25,
        w_risk=0.3,
    ),
}


def get_available_strategies() -> List[BundleStrategyConfig]:
    """Return list of strategies for UI."""
    return list(_STRATEGIES.values())


def get_strategy_config(strategy_name: str) -> BundleStrategyConfig:
    if strategy_name not in _STRATEGIES:
        return _STRATEGIES["balanced"]
    return _STRATEGIES[strategy_name]


# ---------- Core helpers ----------------------------------------------------


def _normalize_series(s: pd.Series) -> pd.Series:
    """Min-max normalize; if degenerate, return 0.5 constant."""
    if s.empty:
        return s
    s = s.astype(float)
    min_val = s.min()
    max_val = s.max()
    if math.isclose(min_val, max_val):
        return pd.Series(0.5, index=s.index)
    return (s - min_val) / (max_val - min_val)


def _extract_genre_set(genres_value: Any) -> set:
    """Convert a genres representation into a set of genre tokens."""
    if genres_value is None or (isinstance(genres_value, float) and math.isnan(genres_value)):
        return set()
    if isinstance(genres_value, (list, tuple, set)):
        return set(str(g).strip().lower() for g in genres_value if g)
    s = str(genres_value)
    if "|" in s:
        parts = s.split("|")
    elif "," in s:
        parts = s.split(",")
    else:
        parts = [s]
    return set(p.strip().lower() for p in parts if p.strip())


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _year_similarity(y1: Optional[int], y2: Optional[int], tau: float = 5.0) -> float:
    if y1 is None or y2 is None:
        return 0.0
    try:
        d = abs(int(y1) - int(y2))
    except ValueError:
        return 0.0
    return math.exp(-d / tau)


def _platform_compatibility(p1: Any, p2: Any) -> float:
    if p1 is None or p2 is None:
        return 0.5
    return 1.0 if str(p1).strip().upper() == str(p2).strip().upper() else 0.3


def _region_compatibility(r1: Any, r2: Any) -> float:
    if r1 is None or r2 is None:
        return 0.5
    return 1.0 if str(r1).strip().upper() == str(r2).strip().upper() else 0.4


def _content_compatibility(row_i: pd.Series, row_j: pd.Series) -> float:
    genres_i = _extract_genre_set(row_i.get("genres"))
    genres_j = _extract_genre_set(row_j.get("genres"))
    g_sim = _jaccard(genres_i, genres_j)

    y_sim = _year_similarity(row_i.get("release_year"), row_j.get("release_year"))
    p_comp = _platform_compatibility(row_i.get("platform"), row_j.get("platform"))
    r_comp = _region_compatibility(row_i.get("region"), row_j.get("region"))

    return 0.5 * g_sim + 0.3 * y_sim + 0.1 * p_comp + 0.1 * r_comp


def _compute_bundle_scores(
        df: pd.DataFrame,
        idxs: Iterable[int],
        price_score: pd.Series,
        fit_score: pd.Series,
        diversity_score: pd.Series,
        risk_score: pd.Series,
        strategy: BundleStrategyConfig,
) -> Tuple[float, float, float, float, float]:
    idxs = list(idxs)
    if not idxs:
        return 0, 0, 0, 0, 0

    bundle_price_score = float(price_score.loc[idxs].mean())
    bundle_fit_score = float(fit_score.loc[idxs].mean())
    bundle_div_score = float(diversity_score.loc[idxs].mean())
    bundle_risk_score = float(risk_score.loc[idxs].mean())

    total_score = (
            strategy.w_value * bundle_price_score
            + strategy.w_fit * bundle_fit_score
            + strategy.w_div * bundle_div_score
            - strategy.w_risk * bundle_risk_score
    )

    return total_score, bundle_price_score, bundle_fit_score, bundle_div_score, bundle_risk_score


# ---------- Public API: scoring + optimization ------------------------------


def prepare_candidate_scores(candidate_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare per-title scores used by the bundling algorithm.

    Expected columns (best effort; missing ones handled gracefully):
        - predicted_price (float)
        - popularity (optional float)
        - release_year (optional int)
        - region (optional str)
        - platform (optional str)
        - genres (optional str / list)
        - risk_proxy (optional float; higher = riskier)
    """
    df = candidate_df.copy()

    if "predicted_price" not in df.columns:
        raise ValueError("candidate_df must contain 'predicted_price' column")

    df["price_score"] = _normalize_series(df["predicted_price"])

    if "popularity" in df.columns:
        df["popularity_score"] = _normalize_series(df["popularity"])
    else:
        df["popularity_score"] = 0.5

    if "release_year" in df.columns:
        df["_age"] = df["release_year"].max() - df["release_year"]
        df["recency_score"] = 1.0 - _normalize_series(df["_age"].fillna(df["_age"].max() or 0))
    else:
        df["recency_score"] = 0.5

    df["fit_score_region"] = 1.0  # assume UI pre-filters; override if needed
    df["fit_score_platform"] = 1.0

    if "risk_proxy" in df.columns:
        df["risk_score"] = _normalize_series(df["risk_proxy"])
    else:
        df["risk_score"] = 0.5

    df["diversity_base"] = 0.5 * df["popularity_score"] + 0.5 * df["recency_score"]

    return df


def generate_candidate_bundles(
        candidate_df: pd.DataFrame,
        bundle_size: int,
        strategy_name: str,
        max_anchors: int = 10,
        max_bundles: int = 10,
) -> List[BundleResult]:
    """
    Main entry point for generating bundles.

    candidate_df must contain at least:
        - id or title_id
        - title_name (for display)
        - predicted_price
        - region, platform, release_year, genres (if available)
    """
    if bundle_size < 2:
        raise ValueError("bundle_size must be at least 2")

    if len(candidate_df) < bundle_size:
        return []

    strategy = get_strategy_config(strategy_name)
    df = prepare_candidate_scores(candidate_df)

    df["anchor_score"] = (
            0.7 * df["price_score"]
            + 0.15 * df["fit_score_region"]
            + 0.15 * df["fit_score_platform"]
    )

    df_sorted = df.sort_values("anchor_score", ascending=False)
    df_sorted = df_sorted.head(max_anchors) if max_anchors > 0 else df_sorted

    price_score = df["price_score"]
    fit_score = (df["fit_score_region"] + df["fit_score_platform"]) / 2.0
    diversity_score = df["diversity_base"]
    risk_score = df["risk_score"]

    bundles: Dict[Tuple[int, ...], BundleResult] = {}

    for anchor_idx in df_sorted.index:
        current_bundle = [anchor_idx]

        while len(current_bundle) < bundle_size:
            best_delta = -1e9
            best_candidate = None

            for idx in df.index:
                if idx in current_bundle:
                    continue
                prev_total, *_ = _compute_bundle_scores(
                    df,
                    current_bundle,
                    price_score,
                    fit_score,
                    diversity_score,
                    risk_score,
                    strategy,
                )
                new_total, *_ = _compute_bundle_scores(
                    df,
                    current_bundle + [idx],
                    price_score,
                    fit_score,
                    diversity_score,
                    risk_score,
                    strategy,
                )
                delta = new_total - prev_total
                if delta > best_delta:
                    best_delta = delta
                    best_candidate = idx

            if best_candidate is None:
                break
            current_bundle.append(best_candidate)

        if len(current_bundle) < 2:
            continue

        key = tuple(sorted(current_bundle))
        if key in bundles:
            continue

        total_score, bundle_price_score, bundle_fit_score, bundle_div_score, bundle_risk_score = (
            _compute_bundle_scores(
                df,
                current_bundle,
                price_score,
                fit_score,
                diversity_score,
                risk_score,
                strategy,
            )
        )

        bundle_titles = []
        raw_prices = []

        for idx in current_bundle:
            row = df.loc[idx]
            title_info = {
                "title_id": row.get("title_id", row.get("id", idx)),
                "title_name": row.get("title_name", row.get("primary_title", f"Title {idx}")),
                "predicted_price": float(row["predicted_price"]),
                "region": row.get("region"),
                "platform": row.get("platform"),
                "release_year": row.get("release_year"),
                "genres": row.get("genres"),
            }
            bundle_titles.append(title_info)
            raw_prices.append(float(row["predicted_price"]))

        bundle_price_raw = float(np.sum(raw_prices))
        rationale = _build_bundle_rationale(bundle_titles, strategy)

        bundles[key] = BundleResult(
            titles=bundle_titles,
            bundle_price_raw=bundle_price_raw,
            bundle_price_score=bundle_price_score,
            bundle_fit_score=bundle_fit_score,
            bundle_diversity_score=bundle_div_score,
            bundle_risk_score=bundle_risk_score,
            total_score=total_score,
            strategy_name=strategy.display_name,
            rationale=rationale,
        )

    results = sorted(bundles.values(), key=lambda b: b.total_score, reverse=True)
    if max_bundles:
        results = results[:max_bundles]
    return results


def _build_bundle_rationale(titles: List[Dict[str, Any]], strategy: BundleStrategyConfig) -> str:
    if not titles:
        return ""

    anchor = max(titles, key=lambda t: t.get("predicted_price", 0.0))

    years = [t.get("release_year") for t in titles if t.get("release_year") is not None]
    year_span = ""
    if years:
        year_min, year_max = min(years), max(years)
        year_span = f" from {year_min}–{year_max}" if year_min != year_max else f" in {year_min}"

    all_genres = []
    for t in titles:
        all_genres.extend(list(_extract_genre_set(t.get("genres"))))
    top_genres = sorted(set(all_genres))
    genres_part = ""
    if top_genres:
        genres_part = f" focusing on {', '.join(g.title() for g in top_genres[:3])}"

    strategy_phrase = {
        "max_value": "maximize overall bundle value",
        "balanced": "balance value and risk",
        "min_risk": "reduce risk while maintaining a diverse catalog",
    }.get(strategy.name, "provide a balanced offering")

    return (
        f"This bundle is anchored by **{anchor.get('title_name', 'a high-value title')}**"
        f"{year_span}{genres_part}, designed to {strategy_phrase} "
        f"for the selected region and platform."
    )
