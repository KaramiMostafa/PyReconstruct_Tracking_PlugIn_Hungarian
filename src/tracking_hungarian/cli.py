from __future__ import annotations
import argparse
from pathlib import Path
import json
from .roi import ROIFrameTable
from .matcher import build_cost_matrix, augmented_hungarian, interpret_assignments
from .tracks import export_lineage_to_mantrack, generate_output_csv

def _load_cfg(path: str) -> dict:
    import yaml
    with open(path,"r") as f: return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser("tracking-hungarian")
    ap.add_argument("--roi", required=True, help="ROI table (.csv or .json)")
    ap.add_argument("--frames", type=int, nargs=2, required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--static-every", type=int, default=1)
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    outdir = Path(args.out or cfg.get("paths",{}).get("output_dir","runs/hungarian_out")); outdir.mkdir(parents=True, exist_ok=True)

    tbl = ROIFrameTable.from_csv(args.roi) if args.roi.endswith(".csv") else ROIFrameTable.from_json(args.roi)
    h = cfg.get("hungarian", {})
    features = tbl.feature_columns(h.get("features"))
    weights = h.get("weights", {"feature":1.0,"spatial":0.2,"area":0.1,"iou":0.5})
    scaling = int(h.get("scaling",{}).get("percentile",95))
    gating = h.get("gating", {"max_spatial":200.0, "min_iou":0.0})
    costs  = h.get("costs", {"birth":0.6,"death":0.6,"max_cost":1e6,"assign_threshold":0.95})

    frames = tbl.frames(); num_frames = int(max(frames)+1) if frames else 0
    tracking = {}; lineage = {}
    for t in range(args.frames[0], args.frames[1]):
        sec1 = tbl.at(t); sec2 = tbl.at(t+1)
        if sec1.empty or sec2.empty: continue
        names1 = [f"{cid}_{t}" for cid in sec1["Label"].tolist()]
        names2 = [f"{cid}_{t+1}" for cid in sec2["Label"].tolist()]
        C = build_cost_matrix(
            sec1, sec2, features,
            weights=weights, scaling_percentile=scaling,
            gating_max_spatial=gating.get("max_spatial"), gating_min_iou=gating.get("min_iou"),
            big_m=float(costs.get("max_cost",1e6))
        )
        rows, cols, C_aug = augmented_hungarian(C, birth_cost=float(costs["birth"]), death_cost=float(costs["death"]), big_m=float(costs["max_cost"]))
        matches, deaths, births = interpret_assignments(rows, cols, C, C_aug, assign_threshold=float(costs["assign_threshold"]))
        # update lineage/tracking (simple 1-1, no divisions)
        # here we just keep a minimal record—plugin can extend as needed
        if t not in lineage: pass
        print(f"[HUN {t}->{t+1}] matches={len(matches)} deaths={len(deaths)} births={len(births)}")
        # You can build tracking timelines similarly to the advanced repo if desired

    # Write placeholders to show outputs exist (you can integrate full lineage build if you want)
    export_lineage_to_mantrack({}, outdir / "man_track.txt")
    generate_output_csv({}, num_frames, outdir / "tracking_results.csv")
