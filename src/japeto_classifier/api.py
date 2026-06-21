from __future__ import annotations

import os
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, field_validator

from japeto_classifier.config import Settings, get_settings
from japeto_classifier.constants import expected_model_ids
from japeto_classifier.registry import ModelRegistry
from japeto_classifier.text import build_model_text


class PredictionRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    model_id: str = Field(min_length=1, max_length=160)
    chatbot_response: str = Field(min_length=1, max_length=20_000)
    user_message: str | None = Field(default=None, max_length=10_000)

    @field_validator("chatbot_response")
    @classmethod
    def response_must_have_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("chatbot_response must not be blank")
        return value


class EmbeddingCache:
    def __init__(self, limit: int = 128) -> None:
        self.limit = limit
        self.values: OrderedDict[tuple[str, str], np.ndarray] = OrderedDict()

    def get(self, model: str, text: str) -> np.ndarray | None:
        key = (model, text)
        value = self.values.get(key)
        if value is not None:
            self.values.move_to_end(key)
        return value

    def put(self, model: str, text: str, value: np.ndarray) -> None:
        key = (model, text)
        self.values[key] = value
        self.values.move_to_end(key)
        while len(self.values) > self.limit:
            self.values.popitem(last=False)


def create_app(settings: Settings | None = None, openai_client: OpenAI | None = None) -> FastAPI:
    configured = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.registry = ModelRegistry(
            configured.registry_path, configured.models_dir, configured.metrics_dir
        )
        app.state.embedding_cache = EmbeddingCache()
        app.state.openai_client = openai_client
        yield

    app = FastAPI(
        title="Japeto Prediction Studio",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def dashboard() -> HTMLResponse:
        if not configured.dashboard_path.exists():
            raise HTTPException(status_code=503, detail="Dashboard HTML is missing")
        return HTMLResponse(configured.dashboard_path.read_text(encoding="utf-8"))

    @app.get("/health")
    def health() -> dict[str, Any]:
        models = app.state.registry.list_models()
        return {
            "status": "ready" if models else "needs_preparation",
            "registered_models": len(models),
            "expected_models": len(expected_model_ids()),
            "openai_configured": bool(os.getenv("OPENAI_API_KEY") or app.state.openai_client),
            "records_available": configured.records_path.exists(),
            "splits_available": configured.splits_path.exists(),
        }

    @app.get("/api/models")
    def list_models() -> dict[str, Any]:
        models = app.state.registry.list_models()
        return {
            "models": models,
            "count": len(models),
            "expected_count": len(expected_model_ids()),
        }

    @app.get("/api/models/{model_id}")
    def model_detail(model_id: str) -> dict[str, Any]:
        try:
            metadata = app.state.registry.get_metadata(model_id)
            metrics = app.state.registry.get_metrics(model_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Model not found") from exc
        return {"metadata": metadata, "metrics": metrics}

    @app.post("/api/predict")
    def predict(request: PredictionRequest) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            metadata = app.state.registry.get_metadata(request.model_id)
            artifact = app.state.registry.load_model(request.model_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Model not found") from exc
        try:
            text = build_model_text(
                request.chatbot_response,
                request.user_message,
                metadata["input_mode"],
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        if metadata["feature_type"] == "tfidf":
            features: Any = [text]
            embedding_cache_hit = None
        else:
            if not (os.getenv("OPENAI_API_KEY") or app.state.openai_client):
                raise HTTPException(
                    status_code=503,
                    detail="OPENAI_API_KEY is required for this model",
                )
            model = metadata["embedding_model"]
            vector = app.state.embedding_cache.get(model, text)
            embedding_cache_hit = vector is not None
            if vector is None:
                try:
                    client = app.state.openai_client or OpenAI()
                    response = client.embeddings.create(
                        model=model,
                        input=[text],
                        encoding_format="float",
                    )
                    vector = np.asarray(response.data[0].embedding, dtype=np.float32)
                except Exception as exc:
                    raise HTTPException(
                        status_code=503,
                        detail="OpenAI could not create an embedding. Check the key, network, and rate limit.",
                    ) from exc
                expected = int(metadata["embedding_dimensions"])
                if vector.shape != (expected,):
                    raise HTTPException(
                        status_code=503,
                        detail=f"Embedding dimension mismatch: expected {expected}",
                    )
                app.state.embedding_cache.put(model, text, vector)
            features = vector.reshape(1, -1)

        estimator = artifact["estimator"]
        probabilities = estimator.predict_proba(features)[0]
        classes = np.asarray(estimator.classes_, dtype=str)
        order = np.argsort(probabilities)[::-1]
        top = [
            {"label": str(classes[index]), "probability": float(probabilities[index])}
            for index in order[:3]
        ]
        return {
            "model_id": request.model_id,
            "prediction": top[0]["label"],
            "top_predictions": top,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "label_scheme": metadata["label_scheme"],
            "input_mode": metadata["input_mode"],
            "feature_type": metadata["feature_type"],
            "embedding_cache_hit": embedding_cache_hit,
        }

    return app


app = create_app()
