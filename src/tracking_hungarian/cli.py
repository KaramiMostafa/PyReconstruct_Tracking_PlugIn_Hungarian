from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .roi import ROIFrameTable
from .pipeline import HungarianConfig, track_series


def _load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser("tracking-hungarian")
    ap.add_argument("--roi", required=True, help="ROI table (.csv or .json)")
    ap.add_argument("--frames", type=int, nargs=2, required=True, help="START END (inclusive)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    outdir = Path(args.out or cfg.get("paths", {}).get("output_dir", "runs/hungarian_out"))
    outdir.mkdir(parents=True, exist_ok=True)

    tbl = ROIFrameTable.from_csv(args.roi) if args.roi.endswith(".csv") else ROIFrameTable.from_json(args.roi)

    h = cfg.get("hungarian", {})
    weights = h.get("weights", {"feature": 1.0, "spatial": 0.2, "area": 0.1, "iou": 0.5})
    scaling = int(h.get("scaling", {}).get("percentile", 95))
    gating = h.get("gating", {"max_spatial": 200.0, "min_iou": 0.0})
    costs = h.get("costs", {"birth": 0.6, "death": 0.6, "max_cost": 1e6, "assign_threshold": 0.95})

    hc = HungarianConfig(
        features=h.get("features"),
        weights=weights,
        scaling_percentile=scaling,
        gating_max_spatial=gating.get("max_spatial"),
        gating_min_iou=gating.get("min_iou"),
        birth_cost=float(costs.get("birth", 0.6)),
        death_cost=float(costs.get("death", 0.6)),
        max_cost=float(costs.get("max_cost", 1e6)),
        assign_threshold=float(costs.get("assign_threshold", 0.95)),
    )

    start_frame, end_frame = int(args.frames[0]), int(args.frames[1])
    tracks_df = track_series(tbl, hc, frame_range=(start_frame, end_frame))
    out_csv = outdir / "tracks.csv"
    tracks_df.to_csv(out_csv, index=False)

    n_tracks = int(tracks_df["TrackID"].nunique()) if not tracks_df.empty else 0
    print(f"[OK] wrote {out_csv}")
    print(f"[OK] rows={len(tracks_df)} tracks={n_tracks}")