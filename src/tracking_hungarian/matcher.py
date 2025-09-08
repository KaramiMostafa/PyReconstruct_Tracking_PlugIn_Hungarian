from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from scipy.optimize import linear_sum_assignment
from .costs import combine_costs

def build_cost_matrix(
    sec1: pd.DataFrame, sec2: pd.DataFrame, feature_cols: List[str],
    *, weights: Dict[str,float], scaling_percentile: int = 95,
    gating_max_spatial: float | None = None, gating_min_iou: float | None = None,
    big_m: float = 1e6
) -> np.ndarray:
    parts = combine_costs(sec1, sec2, feature_cols, weights, scaling_percentile, gating_max_spatial, gating_min_iou)
    C = parts["combined"]
    C = np.where(np.isfinite(C), C, big_m).astype(np.float32)
    return C

def _augment(C: np.ndarray, birth_cost: float, death_cost: float, big_m: float = 1e6):
    n1, n2 = C.shape
    top_right = np.full((n1,n1), big_m, np.float32); np.fill_diagonal(top_right, float(death_cost))
    bottom_left = np.full((n2,n2), big_m, np.float32); np.fill_diagonal(bottom_left, float(birth_cost))
    upper = np.concatenate([C, top_right], axis=1)
    lower = np.concatenate([bottom_left, np.zeros((n2,n1), np.float32)], axis=1)
    return np.concatenate([upper, lower], axis=0)

def augmented_hungarian(C: np.ndarray, birth_cost: float, death_cost: float, big_m: float = 1e6):
    C_aug = _augment(C, birth_cost, death_cost, big_m)
    rows, cols = linear_sum_assignment(C_aug)
    return rows, cols, C_aug

def interpret_assignments(rows, cols, C: np.ndarray, C_aug: np.ndarray, assign_threshold: float):
    n1, n2 = C.shape
    matches, deaths, births = [], [], []
    for r,c in zip(rows, cols):
        if r < n1 and c < n2:
            if C[r,c] <= assign_threshold:
                matches.append((f"1_{r}", f"2_{c}"))
            else:
                deaths.append(r); births.append(c)
        elif r < n1 and c >= n2:
            deaths.append(r)
        elif r >= n1 and c < n2:
            births.append(c)
    return matches, deaths, births
