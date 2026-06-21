from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import Any

import joblib


class ModelRegistry:
    def __init__(self, registry_path: Path, models_dir: Path, metrics_dir: Path) -> None:
        self.registry_path = registry_path
        self.models_dir = models_dir
        self.metrics_dir = metrics_dir
        self._lock = RLock()
        self._cache: dict[str, Any] = {}

    def list_models(self) -> list[dict[str, Any]]:
        if not self.registry_path.exists():
            return []
        payload = json.loads(self.registry_path.read_text(encoding="utf-8"))
        return payload.get("models", [])

    def get_metadata(self, model_id: str) -> dict[str, Any]:
        for item in self.list_models():
            if item["model_id"] == model_id:
                return item
        raise KeyError(model_id)

    def get_metrics(self, model_id: str) -> dict[str, Any]:
        path = self.metrics_dir / f"{model_id}.json"
        if not path.exists():
            raise KeyError(model_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def load_model(self, model_id: str) -> dict[str, Any]:
        with self._lock:
            if model_id not in self._cache:
                path = self.models_dir / f"{model_id}.joblib"
                if not path.exists():
                    raise KeyError(model_id)
                self._cache[model_id] = joblib.load(path)
            return self._cache[model_id]

    def register(
        self,
        model_id: str,
        artifact: dict[str, Any],
        metadata: dict[str, Any],
        metrics: dict[str, Any],
    ) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, self.models_dir / f"{model_id}.joblib", compress=3)
        (self.metrics_dir / f"{model_id}.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        with self._lock:
            models = [item for item in self.list_models() if item["model_id"] != model_id]
            models.append(metadata)
            models.sort(key=lambda item: (-item["metrics"]["macro_f1"], item["model_id"]))
            payload = {
                "champion_model_id": models[0]["model_id"] if models else None,
                "models": models,
            }
            self.registry_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self._cache.pop(model_id, None)

