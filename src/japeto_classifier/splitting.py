from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def _first_fold(
    frame: pd.DataFrame,
    indexes: np.ndarray,
    n_splits: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    subset = frame.iloc[indexes]
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    local_train, local_holdout = next(
        splitter.split(subset, subset["categories"], groups=subset["session_id"])
    )
    return indexes[local_train], indexes[local_holdout]


def create_split_assignments(frame: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("Cannot split an empty dataset")
    all_indexes = np.arange(len(frame))
    remaining, test_indexes = _first_fold(frame, all_indexes, 5, random_state)
    train_indexes, calibration_indexes = _first_fold(frame, remaining, 5, random_state + 1)

    partition = np.full(len(frame), "", dtype=object)
    partition[train_indexes] = "train"
    partition[calibration_indexes] = "calibration"
    partition[test_indexes] = "test"
    assignments = frame[["record_id", "session_id", "categories"]].copy()
    assignments["partition"] = partition
    validate_split_assignments(assignments)
    return assignments[["record_id", "partition"]]


def validate_split_assignments(assignments: pd.DataFrame, records: pd.DataFrame | None = None) -> None:
    if not assignments["record_id"].is_unique:
        raise ValueError("Split assignments contain duplicate record IDs")
    if set(assignments["partition"]) != {"train", "calibration", "test"}:
        raise ValueError("Split assignments must contain train, calibration, and test partitions")
    if records is None:
        return
    merged = records[["record_id", "session_id"]].merge(assignments, on="record_id", validate="one_to_one")
    counts = merged.groupby("session_id")["partition"].nunique()
    if int(counts.max()) > 1:
        raise ValueError("At least one session appears in multiple partitions")


def write_split_assignments(
    records_path: Path,
    destination: Path,
    random_state: int = 42,
) -> dict[str, object]:
    records = pd.read_parquet(records_path)
    assignments = create_split_assignments(records, random_state=random_state)
    validate_split_assignments(assignments, records)
    destination.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_parquet(destination, index=False)
    counts = assignments["partition"].value_counts().to_dict()
    summary = {
        "destination": str(destination),
        "random_state": random_state,
        "counts": {key: int(value) for key, value in counts.items()},
        "grouped_by": "session_id",
    }
    (destination.parent / "splits.manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary

