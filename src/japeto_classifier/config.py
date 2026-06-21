from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    root: Path
    data_dir: Path
    artifacts_dir: Path
    dashboard_path: Path
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    random_state: int = 42

    @property
    def raw_dataset(self) -> Path:
        preferred = self.data_dir / "raw" / "dataset.xlsx"
        legacy = self.root / "dataset.xlsx"
        return preferred if preferred.exists() else legacy

    @property
    def records_path(self) -> Path:
        return self.data_dir / "processed" / "records.parquet"

    @property
    def splits_path(self) -> Path:
        return self.data_dir / "processed" / "splits.parquet"

    @property
    def embeddings_dir(self) -> Path:
        return self.data_dir / "embeddings"

    @property
    def models_dir(self) -> Path:
        return self.artifacts_dir / "models"

    @property
    def metrics_dir(self) -> Path:
        return self.artifacts_dir / "metrics"

    @property
    def registry_path(self) -> Path:
        return self.artifacts_dir / "registry.json"


def get_settings() -> Settings:
    load_dotenv(ROOT / ".env")
    data_dir = Path(os.getenv("JAPETO_DATA_DIR", ROOT / "data"))
    artifacts_dir = Path(os.getenv("JAPETO_ARTIFACTS_DIR", ROOT / "artifacts"))
    dashboard = Path(os.getenv("JAPETO_DASHBOARD_PATH", ROOT / "app" / "dashboard.html"))
    return Settings(ROOT, data_dir.resolve(), artifacts_dir.resolve(), dashboard.resolve())

