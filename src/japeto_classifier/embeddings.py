from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from openai import OpenAI

from japeto_classifier.data import file_sha256
from japeto_classifier.text import build_model_text


def embedding_paths(directory: Path, mode: str) -> tuple[Path, Path, Path]:
    return (
        directory / f"{mode}.npz",
        directory / f"{mode}.manifest.json",
        directory / ".work" / f"{mode}.jsonl",
    )


def _chunks(values: list[tuple[str, str]], size: int) -> Iterable[list[tuple[str, str]]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def _read_checkpoint(path: Path, dimensions: int) -> dict[str, np.ndarray]:
    cached: dict[str, np.ndarray] = {}
    if not path.exists():
        return cached
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            vector = np.asarray(item["embedding"], dtype=np.float32)
            if vector.shape == (dimensions,):
                cached[item["record_id"]] = vector
    return cached


def _request_batch(
    client: OpenAI,
    texts: list[str],
    model: str,
    dimensions: int,
    attempts: int = 6,
) -> list[list[float]]:
    delay = 1.0
    for attempt in range(attempts):
        try:
            response = client.embeddings.create(model=model, input=texts, encoding_format="float")
            vectors = [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
            if any(len(vector) != dimensions for vector in vectors):
                raise ValueError(f"Expected {dimensions}-dimensional embeddings")
            return vectors
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 30)
    raise RuntimeError("Embedding request exhausted retries")


def _embedding_items(records_path: Path, mode: str) -> list[tuple[str, str]]:
    records = pd.read_parquet(records_path)
    if mode == "context_enhanced":
        records = records.loc[records["user_message"].fillna("").str.strip() != ""].copy()
    return [
        (row.record_id, build_model_text(row.chatbot_response, row.user_message, mode))
        for row in records.itertuples(index=False)
    ]


def _input_sha256(items: list[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for record_id, text in items:
        digest.update(record_id.encode("utf-8"))
        digest.update(text.encode("utf-8"))
    return digest.hexdigest()


def embedding_artifact_is_current(
    records_path: Path,
    output_dir: Path,
    mode: str,
    model: str,
    dimensions: int,
) -> bool:
    output_path, manifest_path, _ = embedding_paths(output_dir, mode)
    if not records_path.exists() or not output_path.exists() or not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        items = _embedding_items(records_path, mode)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        manifest.get("model") == model
        and manifest.get("dimensions") == dimensions
        and manifest.get("input_sha256") == _input_sha256(items)
        and manifest.get("rows") == len(items)
    )


def generate_embeddings(
    records_path: Path,
    output_dir: Path,
    mode: str,
    model: str = "text-embedding-3-small",
    dimensions: int = 1536,
    batch_size: int = 64,
    force: bool = False,
    client: OpenAI | None = None,
) -> dict[str, object]:
    if mode not in {"response_only", "context_enhanced"}:
        raise ValueError(f"Unsupported embedding mode: {mode}")
    output_path, manifest_path, checkpoint_path = embedding_paths(output_dir, mode)
    items = _embedding_items(records_path, mode)
    input_sha256 = _input_sha256(items)
    if output_path.exists() and manifest_path.exists() and not force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("model") == model
            and existing.get("dimensions") == dimensions
            and existing.get("input_sha256") == input_sha256
            and existing.get("rows") == len(items)
        ):
            return existing
    if not os.getenv("OPENAI_API_KEY") and client is None:
        raise RuntimeError("OPENAI_API_KEY is required to generate embeddings")

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    cached = {} if force else _read_checkpoint(checkpoint_path, dimensions)
    api = client or OpenAI()

    pending = [(record_id, text) for record_id, text in items if record_id not in cached]
    with checkpoint_path.open("a", encoding="utf-8") as checkpoint:
        for batch in _chunks(pending, batch_size):
            vectors = _request_batch(api, [text for _, text in batch], model, dimensions)
            for (record_id, _), vector in zip(batch, vectors, strict=True):
                cached[record_id] = np.asarray(vector, dtype=np.float32)
                checkpoint.write(json.dumps({"record_id": record_id, "embedding": vector}) + "\n")
            checkpoint.flush()

    record_ids = np.asarray([record_id for record_id, _ in items])
    matrix = np.vstack([cached[record_id] for record_id in record_ids]).astype(np.float32)
    temporary = output_path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, record_ids=record_ids, embeddings=matrix)
    temporary.replace(output_path)

    manifest = {
        "model": model,
        "dimensions": dimensions,
        "input_mode": mode,
        "rows": int(matrix.shape[0]),
        "records_sha256": file_sha256(records_path),
        "input_sha256": input_sha256,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifact": str(output_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    checkpoint_path.unlink(missing_ok=True)
    return manifest


def load_embeddings(path: Path, dimensions: int = 1536) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as artifact:
        record_ids = artifact["record_ids"].astype(str)
        matrix = artifact["embeddings"].astype(np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != dimensions:
        raise ValueError(f"Embedding matrix must have shape (n, {dimensions})")
    if len(record_ids) != len(matrix) or len(set(record_ids)) != len(record_ids):
        raise ValueError("Embedding record IDs must be unique and aligned with the matrix")
    return record_ids, matrix
