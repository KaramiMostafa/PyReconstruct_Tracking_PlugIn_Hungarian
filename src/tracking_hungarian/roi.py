from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union
import json
import pandas as pd

RESERVED = ["FrameID", "Label", "Centroid_X", "Centroid_Y", "Area",
            "ROI_X1","ROI_Y1","ROI_X2","ROI_Y2"]

@dataclass
class ROIFrameTable:
    df: pd.DataFrame

    @classmethod
    def from_csv(cls, path: Union[str, Path]) -> "ROIFrameTable":
        df = pd.read_csv(path)
        return cls._post(df)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "ROIFrameTable":
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict) and "rows" in data: data = data["rows"]
        df = pd.DataFrame(data)
        return cls._post(df)

    @classmethod
    def _post(cls, df: pd.DataFrame) -> "ROIFrameTable":
        required = ["FrameID","Label","Centroid_X","Centroid_Y"]
        for c in required:
            if c not in df.columns: raise ValueError(f"Missing required column: {c}")
        keep = []
        for c in df.columns:
            if c in ["FrameID","Label"]: keep.append(c); continue
            s = df[c].dropna()
            if len(s.unique()) > 1: keep.append(c)
        df = df[keep].sort_values(["FrameID","Label"]).reset_index(drop=True)
        return cls(df)

    def frames(self):
        return sorted(self.df["FrameID"].unique().tolist())

    def at(self, frame: int) -> pd.DataFrame:
        return self.df[self.df["FrameID"] == frame]

    def feature_columns(self, explicit: Optional[List[str]] = None) -> List[str]:
        if explicit:
            for c in explicit:
                if c not in self.df.columns: raise ValueError(f"feature column not found: {c}")
            return explicit
        return [c for c in self.df.columns if c not in RESERVED]
