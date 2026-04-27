# src/tracking_hungarian/pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .roi import ROIFrameTable
from .matcher import build_cost_matrix, augmented_hungarian, interpret_assignments


@dataclass
class HungarianConfig:
    features: Optional[List[str]] = None
    weights: Dict[str, float] = None
    scaling_percentile: int = 95
    gating_max_spatial: Optional[float] = None
    gating_min_iou: Optional[float] = None
    birth_cost: float = 0.6
    death_cost: float = 0.6
    max_cost: float = 1e6
    assign_threshold: float = 0.95


def track_series(
    tbl: ROIFrameTable,
    cfg: HungarianConfig,
    frame_range: Optional[Tuple[int, int]] = None
) -> pd.DataFrame:
    """
    Returns DataFrame with columns: FrameID, Label, TrackID
    """
    if cfg.weights is None:
        cfg.weights = {"feature": 1.0, "spatial": 0.2, "area": 0.1, "iou": 0.5}

    frames = tbl.frames()
    if frame_range is not None:
        a, b = frame_range
        frames = [f for f in frames if a <= f <= b]

    if len(frames) < 1:
        return pd.DataFrame(columns=["FrameID", "Label", "TrackID"])

    # Map: frame -> (row_index_in_that_frame -> TrackID)
    track_map: Dict[int, Dict[int, int]] = {}
    next_tid = 0

    # Init first frame
    sec0 = tbl.at(frames[0]).sort_values("Label").reset_index(drop=True)
    track_map[frames[0]] = {}
    for i in range(len(sec0)):
        track_map[frames[0]][i] = next_tid
        next_tid += 1

    out_rows = []
    for i, r in enumerate(sec0.itertuples()):
        out_rows.append({"FrameID": int(frames[0]), "Label": int(r.Label), "TrackID": int(track_map[frames[0]][i])})

    # Iterate consecutive frames
    for f_prev, f_cur in zip(frames[:-1], frames[1:]):
        sec1 = tbl.at(f_prev).sort_values("Label").reset_index(drop=True)
        sec2 = tbl.at(f_cur).sort_values("Label").reset_index(drop=True)

        if sec1.empty or sec2.empty:
            continue

        feature_cols = tbl.feature_columns(cfg.features)

        C = build_cost_matrix(
            sec1, sec2, feature_cols,
            weights=cfg.weights,
            scaling_percentile=cfg.scaling_percentile,
            gating_max_spatial=cfg.gating_max_spatial,
            gating_min_iou=cfg.gating_min_iou,
            big_m=cfg.max_cost,
        )
        rows, cols, C_aug = augmented_hungarian(C, birth_cost=cfg.birth_cost, death_cost=cfg.death_cost, big_m=cfg.max_cost)
        matches, deaths, births = interpret_assignments(rows, cols, C, C_aug, assign_threshold=cfg.assign_threshold)

        prev_map = track_map[f_prev]
        cur_map: Dict[int, int] = {}

        # Matches
        for a, b in matches:
            i = int(a.split("_")[1])
            j = int(b.split("_")[1])
            cur_map[j] = prev_map[i]

        # Births (unmatched in t+1)
        for j in births:
            cur_map[int(j)] = next_tid
            next_tid += 1

        # Any remaining (safety)
        for j in range(len(sec2)):
            if j not in cur_map:
                cur_map[j] = next_tid
                next_tid += 1

        track_map[f_cur] = cur_map

        for j, r in enumerate(sec2.itertuples()):
            out_rows.append({"FrameID": int(f_cur), "Label": int(r.Label), "TrackID": int(cur_map[j])})

    return pd.DataFrame(out_rows)