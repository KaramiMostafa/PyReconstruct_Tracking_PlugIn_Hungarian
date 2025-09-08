# tracking-hungarian-core

Simple, fast ROI-based tracker:
- Weighted cost matrix (feature, spatial, area, IoU)
- Robust scaling + gating
- Augmented Hungarian (birth/death)
- Exports lineage (man_track.txt) and tracking_results.csv

## Install
```bash
pip install -e .
# optional viz:
pip install -e ".[viz]"
