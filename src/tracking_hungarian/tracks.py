from __future__ import annotations
from pathlib import Path
from typing import Dict
import pandas as pd

def export_lineage_to_mantrack(lineage_info: Dict[int, Dict[str,int]], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for track_id in sorted(lineage_info.keys()):
            L = track_id + 1
            B = lineage_info[track_id]['start']
            E = lineage_info[track_id]['end']
            parent = lineage_info[track_id]['parent']
            P = 0 if parent == 0 else (parent + 1)
            f.write(f"{L} {B} {E} {P}\n")
    print(f"[INFO] man_track.txt: {out_path}")

def generate_output_csv(tracking: Dict[int, list], num_frames: int, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = [[tid] + arr[:num_frames] for tid, arr in tracking.items()]
    cols = ["Tracking_ID"] + [f"Frame_{i}" for i in range(num_frames)]
    pd.DataFrame(rows, columns=cols).to_csv(out_csv, index=False)
    print(f"[INFO] tracking_results.csv: {out_csv}")
