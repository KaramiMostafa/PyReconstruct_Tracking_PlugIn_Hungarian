from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Dict

def _pairwise_euclidean(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    diff = A[:,None,:] - B[None,:,:]
    return np.sqrt(np.sum(diff*diff, axis=-1))

def _pairwise_absdiff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a[:,None] - b[None,:])

def _pairwise_iou(b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    n1, n2 = b1.shape[0], b2.shape[0]
    out = np.zeros((n1,n2), np.float32)
    for i in range(n1):
        x1a,y1a,x2a,y2a = b1[i]
        for j in range(n2):
            x1b,y1b,x2b,y2b = b2[j]
            xi1, yi1 = max(x1a,x1b), max(y1a,y1b)
            xi2, yi2 = min(x2a,x2b), min(y2a,y2b)
            w,h = max(0.0, xi2-xi1), max(0.0, yi2-yi1)
            inter = w*h
            area_a = max(0.0, x2a-x1a)*max(0.0, y2a-y1a)
            area_b = max(0.0, x2b-x1b)*max(0.0, y2b-y1b)
            union = area_a + area_b - inter
            out[i,j] = (inter/union) if union>0 else 0.0
    return out

def iou_1m(b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    return 1.0 - _pairwise_iou(b1,b2)

def combine_costs(
    sec1: pd.DataFrame, sec2: pd.DataFrame, feature_cols: list,
    weights: Dict[str,float], scaling_percentile: int = 95,
    gating_max_spatial: Optional[float] = None, gating_min_iou: Optional[float] = None
) -> Dict[str, np.ndarray]:
    n1, n2 = len(sec1), len(sec2)
    out = {"feature": None, "spatial": None, "area": None, "iou": None, "combined": None}

    # feature
    if feature_cols:
        A = sec1[feature_cols].to_numpy(np.float32)
        B = sec2[feature_cols].to_numpy(np.float32)
        feat = _pairwise_euclidean(A,B)
        s = np.percentile(feat, scaling_percentile) or 1.0
        feat_s = feat/(s+1e-8) * float(weights.get("feature",1.0))
    else:
        feat_s = np.zeros((n1,n2), np.float32)

    # spatial
    dx = _pairwise_absdiff(sec1["Centroid_X"].to_numpy(), sec2["Centroid_X"].to_numpy())
    dy = _pairwise_absdiff(sec1["Centroid_Y"].to_numpy(), sec2["Centroid_Y"].to_numpy())
    spatial = np.sqrt(dx*dx + dy*dy)
    s_sp = np.percentile(spatial, scaling_percentile) or 1.0
    spatial_s = spatial/(s_sp+1e-8) * float(weights.get("spatial",0.0))

    # area
    if "Area" in sec1.columns and "Area" in sec2.columns:
        a1 = sec1["Area"].to_numpy(np.float32); a2 = sec2["Area"].to_numpy(np.float32)
        area = _pairwise_absdiff(a1,a2)/(a1[:,None]+1e-8)
        s_ar = np.percentile(area, scaling_percentile) or 1.0
        area_s = area/(s_ar+1e-8) * float(weights.get("area",0.0))
    else:
        area_s = np.zeros((n1,n2), np.float32)

    # IoU
    has_bbox = all(c in sec1.columns for c in ["ROI_X1","ROI_Y1","ROI_X2","ROI_Y2"]) and \
               all(c in sec2.columns for c in ["ROI_X1","ROI_Y1","ROI_X2","ROI_Y2"])
    if has_bbox:
        b1 = sec1[["ROI_X1","ROI_Y1","ROI_X2","ROI_Y2"]].to_numpy(np.float32)
        b2 = sec2[["ROI_X1","ROI_Y1","ROI_X2","ROI_Y2"]].to_numpy(np.float32)
        iou_c = iou_1m(b1,b2)
        s_iou = np.percentile(iou_c, scaling_percentile) or 1.0
        iou_s = iou_c/(s_iou+1e-8) * float(weights.get("iou",0.0))
    else:
        iou_s = np.zeros((n1,n2), np.float32)

    combined = feat_s + spatial_s + area_s + iou_s

    # gating
    if gating_max_spatial is not None:
        combined = np.where(spatial <= gating_max_spatial, combined, np.inf)
    # Optional: gating by IoU threshold could be added similarly (recompute IoU)

    out.update({"feature": feat_s, "spatial": spatial_s, "area": area_s, "iou": iou_s, "combined": combined})
    return out
