import numpy as np
import pandas as pd

from tracking_hungarian.matcher import (
    build_cost_matrix,
    augmented_hungarian,
    interpret_assignments,
)


def _pair_frames_basic():
    # Two near-perfect correspondences
    s1 = pd.DataFrame({
        "FrameID":[0,0],
        "Label":[1,2],
        "Centroid_X":[0.0, 10.0],
        "Centroid_Y":[0.0,  0.0],
        "Area":[100, 120],
        "F_a":[0.10, 0.20],
        "F_b":[1.00, 0.90],
    })
    s2 = pd.DataFrame({
        "FrameID":[1,1],
        "Label":[3,4],
        "Centroid_X":[0.2, 10.1],
        "Centroid_Y":[0.1, -0.1],
        "Area":[101, 119],
        "F_a":[0.11, 0.19],
        "F_b":[1.02, 0.91],
    })
    feats = ["F_a","F_b"]
    return s1, s2, feats


def test_augmented_hungarian_basic_match():
    s1, s2, feats = _pair_frames_basic()
    C = build_cost_matrix(
        s1, s2, feats,
        weights={"feature":1.0, "spatial":0.5, "area":0.1, "iou":0.0},
        scaling_percentile=100, gating_max_spatial=None, gating_min_iou=None, big_m=1e6
    )
    rows, cols, C_aug = augmented_hungarian(C, birth_cost=0.6, death_cost=0.6, big_m=1e6)
    matches, deaths, births = interpret_assignments(rows, cols, C, C_aug, assign_threshold=0.9)

    # Expect two 1-1 matches; indices should be aligned: 0->0 and 1->1
    assert len(matches) == 2
    assert ("1_0", "2_0") in matches
    assert ("1_1", "2_1") in matches
    assert deaths == []
    assert births == []


def test_birth_death_when_missing_partner():
    # Two parents at t, only one child at t+1 (the other disappears)
    s1 = pd.DataFrame({
        "FrameID":[0,0],
        "Label":[1,2],
        "Centroid_X":[0.0, 10.0],
        "Centroid_Y":[0.0,  0.0],
        "Area":[100, 120],
        "F_a":[0.10, 0.20],
        "F_b":[1.00, 0.90],
    })
    s2 = pd.DataFrame({
        "FrameID":[1],
        "Label":[3],
        "Centroid_X":[10.1],
        "Centroid_Y":[-0.1],
        "Area":[119],
        "F_a":[0.19],
        "F_b":[0.91],
    })
    feats = ["F_a","F_b"]

    C = build_cost_matrix(
        s1, s2, feats,
        weights={"feature":1.0, "spatial":0.5, "area":0.1, "iou":0.0},
        scaling_percentile=100, gating_max_spatial=None, gating_min_iou=None, big_m=1e6
    )
    rows, cols, C_aug = augmented_hungarian(C, birth_cost=0.6, death_cost=0.6, big_m=1e6)
    matches, deaths, births = interpret_assignments(rows, cols, C, C_aug, assign_threshold=0.9)

    # Expect exactly one match to child j=0 (closest to parent i=1),
    # and one death for the unmatched parent i=0. No births (no extra child).
    assert ("1_1", "2_0") in matches
    assert len(matches) == 1
    assert deaths == [0] or deaths == [0,]  # the first parent disappears
    assert births == []  # no extra child appears
