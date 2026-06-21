from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from japeto_classifier.config import Settings
from japeto_classifier.constants import (
    ALGORITHMS,
    FEATURE_TYPES,
    GROUPED_LABELS,
    INPUT_MODES,
    LABEL_SCHEMES,
    ORIGINAL_LABELS,
)
from japeto_classifier.data import file_sha256
from japeto_classifier.embeddings import embedding_paths, load_embeddings
from japeto_classifier.registry import ModelRegistry
from japeto_classifier.splitting import validate_split_assignments
from japeto_classifier.text import build_model_text


@dataclass(frozen=True)
class ModelSpec:
    algorithm: str
    feature_type: str
    label_scheme: str
    input_mode: str

    @property
    def model_id(self) -> str:
        return f"{self.algorithm}__{self.feature_type}__{self.label_scheme}__{self.input_mode}"


def model_specs(features: str = "all") -> list[ModelSpec]:
    feature_values = FEATURE_TYPES if features == "all" else (features,)
    specs: list[ModelSpec] = []
    for mode in INPUT_MODES:
        for labels in LABEL_SCHEMES:
            for feature in feature_values:
                for algorithm in ALGORITHMS:
                    if feature == "openai" and algorithm == "naive_bayes":
                        continue
                    specs.append(ModelSpec(algorithm, feature, labels, mode))
    return specs


def _estimator_and_grid(spec: ModelSpec, random_state: int, quick: bool) -> tuple[BaseEstimator, dict[str, list[Any]]]:
    if spec.algorithm == "svm":
        classifier: BaseEstimator = SVC(random_state=random_state)
        grid = {"C": [1, 10], "kernel": ["linear", "poly"], "degree": [2], "gamma": ["scale"]}
    elif spec.algorithm == "random_forest":
        classifier = RandomForestClassifier(random_state=random_state, n_jobs=-1)
        grid = {
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
        }
    elif spec.algorithm == "naive_bayes":
        classifier = MultinomialNB()
        grid = {"alpha": [0.1, 0.5, 1.0, 1.5]}
    else:
        raise ValueError(f"Unsupported algorithm: {spec.algorithm}")

    if quick:
        grid = {key: [values[0]] for key, values in grid.items()}
    if spec.feature_type == "tfidf":
        estimator = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        strip_accents="unicode",
                        stop_words="english",
                        max_features=1500,
                        ngram_range=(1, 2),
                        sublinear_tf=True,
                    ),
                ),
                ("classifier", classifier),
            ]
        )
        grid = {f"classifier__{key}": values for key, values in grid.items()}
        return estimator, grid
    return classifier, grid


def _calibrate_prefit(estimator: BaseEstimator, x: Any, y: np.ndarray) -> BaseEstimator:
    try:
        from sklearn.frozen import FrozenEstimator

        # FrozenEstimator prevents refitting the already-trained classifier.
        # sklearn still uses a stratified CV splitter to obtain its predictions,
        # so cap the folds at the smallest calibration-class count. The original
        # labels have only two calibration examples in the rarest class.
        _, class_counts = np.unique(y, return_counts=True)
        calibration_folds = min(5, int(class_counts.min()))
        if calibration_folds < 2:
            raise ValueError(
                "Probability calibration needs at least two calibration records "
                "for every class. Regenerate the split or collect more examples."
            )
        calibrated = CalibratedClassifierCV(
            FrozenEstimator(estimator),
            method="sigmoid",
            cv=calibration_folds,
        )
    except ImportError:
        calibrated = CalibratedClassifierCV(estimator, method="sigmoid", cv="prefit")
    calibrated.fit(x, y)
    return calibrated


def _class_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> list[dict[str, Any]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = cm.sum()
    rows: list[dict[str, Any]] = []
    for index, label in enumerate(labels):
        tp = cm[index, index]
        fn = cm[index, :].sum() - tp
        fp = cm[:, index].sum() - tp
        tn = total - tp - fn - fp
        specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
        fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0
        rows.append(
            {
                "label": label,
                "support": int(support[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "specificity": specificity,
                "false_positive_rate": fpr,
            }
        )
    return rows


def _prepare_frame(settings: Settings, spec: ModelSpec) -> pd.DataFrame:
    records = pd.read_parquet(settings.records_path)
    splits = pd.read_parquet(settings.splits_path)
    validate_split_assignments(splits, records)
    frame = records.merge(splits, on="record_id", validate="one_to_one")
    if spec.input_mode == "context_enhanced":
        frame = frame.loc[frame["user_message"].fillna("").str.strip() != ""].copy()
    target = "categories" if spec.label_scheme == "original" else "category_grouped"
    frame["target"] = frame[target]
    frame["model_text"] = [
        build_model_text(row.chatbot_response, row.user_message, spec.input_mode)
        for row in frame.itertuples(index=False)
    ]
    return frame


def _feature_values(settings: Settings, spec: ModelSpec, frame: pd.DataFrame) -> Any:
    if spec.feature_type == "tfidf":
        return frame["model_text"].to_numpy(dtype=object)
    path, _, _ = embedding_paths(settings.embeddings_dir, spec.input_mode)
    if not path.exists():
        raise FileNotFoundError(f"Embedding artifact missing for {spec.input_mode}: {path}")
    record_ids, matrix = load_embeddings(path, settings.embedding_dimensions)
    positions = {record_id: index for index, record_id in enumerate(record_ids)}
    missing = [record_id for record_id in frame["record_id"] if record_id not in positions]
    if missing:
        raise ValueError(f"Embedding artifact is missing {len(missing)} records")
    return matrix[[positions[record_id] for record_id in frame["record_id"]]]


def _select(values: Any, mask: np.ndarray) -> Any:
    return values[mask]


def train_model(settings: Settings, spec: ModelSpec, quick: bool = False) -> dict[str, Any]:
    frame = _prepare_frame(settings, spec)
    values = _feature_values(settings, spec, frame)
    targets = frame["target"].to_numpy(dtype=str)
    partitions = frame["partition"].to_numpy(dtype=str)
    groups = frame["session_id"].to_numpy(dtype=str)
    train_mask = partitions == "train"
    calibration_mask = partitions == "calibration"
    test_mask = partitions == "test"
    if not train_mask.any() or not calibration_mask.any() or not test_mask.any():
        raise ValueError("Every model needs train, calibration, and test records")

    labels = ORIGINAL_LABELS if spec.label_scheme == "original" else GROUPED_LABELS
    estimator, param_grid = _estimator_and_grid(spec, settings.random_state, quick)
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=settings.random_state)
    search = GridSearchCV(
        estimator,
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )
    started = time.perf_counter()
    search.fit(
        _select(values, train_mask),
        targets[train_mask],
        groups=groups[train_mask],
    )
    calibrated = _calibrate_prefit(
        search.best_estimator_,
        _select(values, calibration_mask),
        targets[calibration_mask],
    )
    predictions = calibrated.predict(_select(values, test_mask))
    probabilities = calibrated.predict_proba(_select(values, test_mask))
    probability_labels = list(calibrated.classes_)
    duration = time.perf_counter() - started

    y_test = targets[test_mask]
    accuracy = float(accuracy_score(y_test, predictions))
    _, _, macro_f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_test, predictions, labels=labels)
    test_log_loss = float(log_loss(y_test, probabilities, labels=probability_labels))
    class_rows = _class_metrics(y_test, predictions, labels)

    metrics = {
        "model_id": spec.model_id,
        "algorithm": spec.algorithm,
        "feature_type": spec.feature_type,
        "label_scheme": spec.label_scheme,
        "input_mode": spec.input_mode,
        "accuracy": accuracy,
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "log_loss": test_log_loss,
        "cv_macro_f1_mean": float(search.best_score_),
        "cv_macro_f1_std": float(search.cv_results_["std_test_score"][search.best_index_]),
        "best_params": search.best_params_,
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "class_metrics": class_rows,
        "classification_report": classification_report(
            y_test, predictions, labels=labels, output_dict=True, zero_division=0
        ),
        "partition_counts": {
            "train": int(train_mask.sum()),
            "calibration": int(calibration_mask.sum()),
            "test": int(test_mask.sum()),
        },
        "training_seconds": duration,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "records_sha256": file_sha256(settings.records_path),
        "splits_sha256": file_sha256(settings.splits_path),
    }
    metadata = {
        "model_id": spec.model_id,
        "algorithm": spec.algorithm,
        "feature_type": spec.feature_type,
        "label_scheme": spec.label_scheme,
        "input_mode": spec.input_mode,
        "classes": probability_labels,
        "embedding_model": settings.embedding_model if spec.feature_type == "openai" else None,
        "embedding_dimensions": settings.embedding_dimensions if spec.feature_type == "openai" else None,
        "best_params": search.best_params_,
        "metrics": {
            key: metrics[key]
            for key in ["accuracy", "macro_f1", "weighted_f1", "log_loss", "cv_macro_f1_mean", "cv_macro_f1_std"]
        },
        "partition_counts": metrics["partition_counts"],
        "evaluated_at": metrics["evaluated_at"],
    }
    artifact = {
        "model_id": spec.model_id,
        "estimator": calibrated,
        "feature_type": spec.feature_type,
        "input_mode": spec.input_mode,
        "label_scheme": spec.label_scheme,
        "classes": probability_labels,
        "embedding_model": metadata["embedding_model"],
        "embedding_dimensions": metadata["embedding_dimensions"],
    }
    registry = ModelRegistry(settings.registry_path, settings.models_dir, settings.metrics_dir)
    registry.register(spec.model_id, artifact, metadata, metrics)
    return metadata


def train_catalogue(
    settings: Settings,
    features: str = "all",
    quick: bool = False,
    only: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    selected = set(only or [])
    results: list[dict[str, Any]] = []
    for spec in model_specs(features):
        if selected and spec.model_id not in selected:
            continue
        results.append(train_model(settings, spec, quick=quick))
    return results


def evaluation_summary(settings: Settings) -> dict[str, Any]:
    registry = ModelRegistry(settings.registry_path, settings.models_dir, settings.metrics_dir)
    models = registry.list_models()
    return {
        "champion_model_id": models[0]["model_id"] if models else None,
        "model_count": len(models),
        "models": models,
    }
