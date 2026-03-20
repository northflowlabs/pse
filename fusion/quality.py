"""
PSE data quality scoring and conflict resolution for multi-source fusion.

When two sources provide the same variable, the fusion engine uses quality
scores to compute a weighted blend.  This module provides:

  - Per-source quality weights based on DataQuality objects
  - Conflict detection (large inter-source discrepancies)
  - Quality-weighted averaging
  - Provenance tracking for fused outputs
"""
from __future__ import annotations

import numpy as np
import xarray as xr


def compute_weights(
    quality_scores: list[float],
    prefer_recent: bool = True,
    recency_scores: list[float] | None = None,
) -> np.ndarray:
    """
    Compute normalised quality weights from a list of quality scores.

    Args:
        quality_scores:  Overall quality score (0–1) for each source.
        prefer_recent:   If True, blend recency_scores into weights.
        recency_scores:  Recency score (0–1, higher = more recent) per source.

    Returns:
        Numpy array of weights that sum to 1.
    """
    q = np.array(quality_scores, dtype=np.float64)
    q = np.clip(q, 0.0, 1.0)

    if prefer_recent and recency_scores is not None:
        r = np.clip(np.array(recency_scores, dtype=np.float64), 0.0, 1.0)
        combined = 0.7 * q + 0.3 * r
    else:
        combined = q

    total = combined.sum()
    if total == 0:
        return np.ones(len(q)) / len(q)
    return combined / total


def quality_weighted_merge(
    datasets: list[xr.Dataset],
    weights: np.ndarray,
) -> xr.Dataset:
    """
    Merge multiple aligned Datasets using quality-weighted averaging.

    For each variable present in more than one Dataset, the output is:
      merged = sum(weight_i * ds_i[var]) for all i where var is present

    Variables present in only one Dataset are passed through unchanged.

    Args:
        datasets:  List of spatiotemporally aligned xarray.Datasets.
        weights:   Per-dataset weight array (must sum to 1).

    Returns:
        Merged Dataset with a 'pse_weights' attribute documenting the blend.
    """
    if not datasets:
        raise ValueError("Cannot merge empty dataset list")

    if len(datasets) == 1:
        return datasets[0]

    # Collect all variable names across all datasets
    all_vars: dict[str, list[tuple[int, xr.DataArray]]] = {}
    for i, ds in enumerate(datasets):
        for var in ds.data_vars:
            all_vars.setdefault(var, []).append((i, ds[var]))

    merged_vars: dict[str, xr.DataArray] = {}
    for var, sources in all_vars.items():
        if len(sources) == 1:
            # Only one source — pass through
            merged_vars[var] = sources[0][1]
        else:
            # Quality-weighted blend
            source_weights = np.array([weights[i] for i, _ in sources])
            source_weights = source_weights / source_weights.sum()  # renormalise

            stack = [da.values.astype(np.float32) for _, da in sources]
            blended = np.zeros_like(stack[0], dtype=np.float32)
            total_weight = np.zeros_like(stack[0], dtype=np.float32)

            for da_arr, w in zip(stack, source_weights):
                valid_mask = ~np.isnan(da_arr)
                blended = np.where(valid_mask, blended + w * da_arr, blended)
                total_weight = np.where(valid_mask, total_weight + w, total_weight)

            # Normalise where we had at least some valid data
            with np.errstate(invalid="ignore", divide="ignore"):
                result = np.where(total_weight > 0, blended / total_weight, np.nan)

            # Carry dims/coords from the first source
            ref_da = sources[0][1]
            merged_vars[var] = xr.DataArray(result, dims=ref_da.dims, coords=ref_da.coords)

    # Build output dataset from the first dataset's coordinates
    ref_ds = datasets[0]
    out = xr.Dataset(merged_vars, coords=ref_ds.coords, attrs=ref_ds.attrs)
    out.attrs["pse_fusion_weights"] = weights.tolist()
    return out


def detect_conflicts(
    datasets: list[xr.Dataset],
    variable: str,
    threshold: float = 0.3,
) -> dict:
    """
    Detect large inter-source disagreements for a variable.

    Returns a dict describing any conflicts found.  A conflict is flagged
    when the coefficient of variation across sources exceeds *threshold*.
    """
    arrays = []
    for ds in datasets:
        if variable in ds:
            arr = ds[variable].values.astype(np.float64)
            arr = arr[~np.isnan(arr)]
            if arr.size > 0:
                arrays.append(arr.mean())

    if len(arrays) < 2:
        return {"conflict": False, "sources": len(arrays)}

    mean_val = np.mean(arrays)
    std_val = np.std(arrays)
    cv = std_val / abs(mean_val) if abs(mean_val) > 1e-9 else 0.0

    return {
        "conflict": cv > threshold,
        "variable": variable,
        "coefficient_of_variation": round(cv, 4),
        "source_means": [round(v, 4) for v in arrays],
        "overall_mean": round(mean_val, 4),
        "threshold": threshold,
    }
