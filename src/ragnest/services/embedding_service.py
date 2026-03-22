"""Embedding providers — ABC + Ollama implementation."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import ollama

from ragnest.exceptions import EmbeddingError

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers.

    Each provider wraps a single model. The ``EmbeddingService`` factory
    creates and caches providers per model name.
    """

    @abstractmethod
    def embed_batch(
        self, texts: list[str], batch_size: int = 32
    ) -> list[list[float]]:
        """Embed a list of texts. Implementations should handle internal batching."""
        ...

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        ...


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Ollama-backed embedding provider for a single model."""

    def __init__(self, base_url: str, model: str) -> None:
        self._client = ollama.Client(host=base_url)
        self._model = model
        self._available = False

    def ensure_model(self) -> None:
        """Check model availability, pull if missing."""
        if self._available:
            return
        try:
            self._client.show(self._model)
            self._available = True
        except ollama.ResponseError:
            logger.info("Pulling embedding model '%s'...", self._model)
            try:
                self._client.pull(self._model)
                self._available = True
            except ollama.ResponseError as exc:
                msg = f"Failed to pull model '{self._model}': {exc}"
                raise EmbeddingError(msg) from exc

    def embed_batch(
        self, texts: list[str], batch_size: int = 32
    ) -> list[list[float]]:
        """Embed texts in batches to limit memory usage."""
        self.ensure_model()
        results: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                response = self._client.embed(
                    model=self._model, input=batch
                )
                results.extend(
                    [list(seq) for seq in response.embeddings]
                )
            except (ollama.ResponseError, KeyError) as exc:
                msg = (
                    f"Embedding failed for model '{self._model}' "
                    f"(batch {i // batch_size}): {exc}"
                )
                raise EmbeddingError(msg) from exc
        return results

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        self.ensure_model()
        try:
            response = self._client.embed(
                model=self._model, input=[query]
            )
            return list(response.embeddings[0])
        except (ollama.ResponseError, KeyError, IndexError) as exc:
            msg = f"Query embedding failed for model '{self._model}': {exc}"
            raise EmbeddingError(msg) from exc


class EmbeddingService:
    """Factory that creates and caches providers per model.

    Usage::

        svc = EmbeddingService("http://localhost:11434")
        provider = svc.get_provider("bge-m3")
        vectors = provider.embed_batch(["hello", "world"])
    """

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url
        self._providers: dict[str, OllamaEmbeddingProvider] = {}
        self._client = ollama.Client(host=base_url)

    def get_provider(self, model: str) -> EmbeddingProvider:
        """Return a cached provider for *model*, creating one if needed."""
        if model not in self._providers:
            self._providers[model] = OllamaEmbeddingProvider(
                self._base_url, model
            )
        return self._providers[model]

    def list_models(self) -> list[str]:
        """List all models available in Ollama."""
        try:
            response = self._client.list()
            models: list[Any] = list(  # pyright: ignore[reportUnknownArgumentType]
                response.models  # pyright: ignore[reportUnknownMemberType]
            )
            return [
                str(m.model)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
                for m in models
            ]
        except (ollama.ResponseError, AttributeError) as exc:
            logger.warning("Failed to list Ollama models: %s", exc)
            return []
