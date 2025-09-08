import numpy as np
import pandas as pd

from tracking_hungarian.costs import combine_costs, iou_1m


def _toy_frames():
    # Two objects in t (frame 0) and two in t+1 (frame 1)
    sec1 = pd.DataFrame({
        "FrameID":   [0, 0],
        "Label":     [1, 2],
        "Centroid_X":[0.0, 10.0],
        "Centroid_Y":[0.0,  0.0],
        "Area":      [100, 120],
        "ROI_X1":    [0.0, 10.0],
        "ROI_Y1":    [0.0,  0.0],
        "ROI_X2":    [10.0, 20.0],
        "ROI_Y2":    [10.0, 10.0],
        "F_a":       [0.10, 0.20],
        "F_b":       [1.00, 0.90],
    })
    sec2 = pd.DataFrame({
        "FrameID":   [1, 1],
        "Label":     [3, 4],
        "Centroid_X":[0.5, 10.1],
        "Centroid_Y":[0.2, -0.1],
        "Area":      [101, 119],
        "ROI_X1":    [0.5, 10.1],
        "ROI_Y1":    [0.2, -0.1],
        "ROI_X2":    [10.5, 20.1],
        "ROI_Y2":    [10.2,  9.9],
        "F_a":       [0.11, 0.19],
        "F_b":       [1.02, 0.91],
    })
    return sec1, sec2


def test_combine_costs_shapes_and_values():
    sec1, sec2 = _toy_frames()
    features = ["F_a", "F_b"]
    out = combine_costs(
        sec1, sec2, features,
        weights={"feature": 1.0, "spatial": 0.2, "area": 0.1, "iou": 0.5},
        scaling_percentile=100,  # robust but stable for tiny test
        gating_max_spatial=None,
        gating_min_iou=None,
    )

    # All expected keys present
    for k in ("feature", "spatial", "area", "iou", "combined"):
        assert k in out, f"missing key {k}"

    # Shapes correct and finite
    for k, mat in out.items():
        assert isinstance(mat, np.ndarray)
        assert mat.shape == (2, 2)
        assert np.all(np.isfinite(mat) | np.isinf(mat))

    # Combined ≈ weighted sum of components (no gating here)
    approx_sum = out["feature"] + out["spatial"] + out["area"] + out["iou"]
    assert np.allclose(out["combined"], approx_sum, atol=1e-6)


def test_gating_by_spatial_distance_sets_inf():
    sec1, sec2 = _toy_frames()
    features = ["F_a", "F_b"]
    out = combine_costs(
        sec1, sec2, features,
        weights={"feature": 1.0, "spatial": 1.0, "area": 0.0, "iou": 0.0},
        scaling_percentile=100,
        gating_max_spatial=1.0,   # cross pairs ~10 px apart should be gated
        gating_min_iou=None,
    )
    C = out["combined"]
    # (0->1) and (1->0) should be invalid (inf), diagonals should be finite
    assert np.isfinite(C[0, 0]) and np.isfinite(C[1, 1])
    assert np.isinf(C[0, 1]) and np.isinf(C[1, 0])


def test_iou_1m_identity_and_disjoint():
    # identical boxes → IoU=1 ⇒ 1-IoU=0
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b = np.array([[0, 0, 10, 10]], dtype=np.float32)
    cost = iou_1m(a, b)
    assert cost.shape == (1, 1)
    assert np.isclose(cost[0, 0], 0.0, atol=1e-6)

    # disjoint boxes → IoU=0 ⇒ 1-IoU=1
    b2 = np.array([[20, 20, 30, 30]], dtype=np.float32)
    cost2 = iou_1m(a, b2)
    assert np.isclose(cost2[0, 0], 1.0, atol=1e-6)
