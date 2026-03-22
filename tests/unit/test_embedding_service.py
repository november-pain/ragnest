"""Unit tests for EmbeddingService and OllamaEmbeddingProvider with mocked Ollama client."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import ollama
import pytest

from ragnest.exceptions import EmbeddingError
from ragnest.services.embedding_service import (
    EmbeddingService,
    OllamaEmbeddingProvider,
)


@pytest.fixture
def mock_ollama_client() -> MagicMock:
    """Mock ollama.Client with common methods."""
    client = MagicMock()
    client.show.return_value = {"name": "bge-m3"}
    return client


@pytest.fixture
def provider(mock_ollama_client: MagicMock) -> OllamaEmbeddingProvider:
    """OllamaEmbeddingProvider with injected mock client."""
    p = OllamaEmbeddingProvider.__new__(OllamaEmbeddingProvider)
    p._client = mock_ollama_client  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    p._model = "bge-m3"  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    p._available = False  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    return p


# -- embed_batch --


class TestEmbedBatch:
    """OllamaEmbeddingProvider.embed_batch returns list of embeddings."""

    def test_single_batch(
        self,
        provider: OllamaEmbeddingProvider,
        mock_ollama_client: MagicMock,
    ) -> None:
        embedding = [0.1] * 1024
        mock_ollama_client.embed.return_value = SimpleNamespace(
            embeddings=[embedding],
        )

        results = provider.embed_batch(["hello"])

        assert len(results) == 1
        assert results[0] == embedding
        mock_ollama_client.embed.assert_called_once()

    def test_multiple_items_in_single_batch(
        self,
        provider: OllamaEmbeddingProvider,
        mock_ollama_client: MagicMock,
    ) -> None:
        emb1 = [0.1] * 1024
        emb2 = [0.2] * 1024
        mock_ollama_client.embed.return_value = SimpleNamespace(
            embeddings=[emb1, emb2],
        )

        results = provider.embed_batch(["hello", "world"])

        expected_count = 2
        assert len(results) == expected_count
        assert results[0] == emb1
        assert results[1] == emb2

    def test_large_input_splits_into_batches(
        self,
        provider: OllamaEmbeddingProvider,
        mock_ollama_client: MagicMock,
    ) -> None:
        """33 items with batch_size=32 should produce 2 API calls."""
        emb = [0.1] * 1024
        batch_size = 32
        total_items = 33
        # First call returns 32 embeddings, second returns 1
        mock_ollama_client.embed.side_effect = [
            SimpleNamespace(embeddings=[emb] * batch_size),
            SimpleNamespace(embeddings=[emb]),
        ]

        results = provider.embed_batch(
            [f"text_{i}" for i in range(total_items)], batch_size=batch_size,
        )

        assert len(results) == total_items
        expected_calls = 2
        assert mock_ollama_client.embed.call_count == expected_calls

    def test_empty_input_returns_empty_list(
        self,
        provider: OllamaEmbeddingProvider,
    ) -> None:
        results = provider.embed_batch([])

        assert results == []


# -- embed_query --


class TestEmbedQuery:
    """OllamaEmbeddingProvider.embed_query returns single embedding."""

    def test_returns_single_embedding(
        self,
        provider: OllamaEmbeddingProvider,
        mock_ollama_client: MagicMock,
    ) -> None:
        embedding = [0.5] * 1024
        mock_ollama_client.embed.return_value = SimpleNamespace(
            embeddings=[embedding],
        )

        result = provider.embed_query("test query")

        assert result == embedding
        mock_ollama_client.embed.assert_called_once_with(
            model="bge-m3", input=["test query"],
        )


# -- ensure_model --


class TestEnsureModel:
    """OllamaEmbeddingProvider.ensure_model caches after first call."""

    def test_caches_after_first_call(
        self,
        provider: OllamaEmbeddingProvider,
        mock_ollama_client: MagicMock,
    ) -> None:
        provider.ensure_model()
        provider.ensure_model()

        # show() should only be called once — second call is cached
        mock_ollama_client.show.assert_called_once_with("bge-m3")

    def test_sets_available_flag(
        self,
        provider: OllamaEmbeddingProvider,
    ) -> None:
        provider.ensure_model()

        assert provider._available is True  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]


# -- Connection failure --


class TestConnectionFailure:
    """Connection errors raise EmbeddingError."""

    def test_embed_query_raises_on_connection_error(
        self,
        provider: OllamaEmbeddingProvider,
        mock_ollama_client: MagicMock,
    ) -> None:
        # Make ensure_model pass
        provider._available = True  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        mock_ollama_client.embed.side_effect = ollama.ResponseError("connection refused")

        with pytest.raises(EmbeddingError, match="Query embedding failed"):
            provider.embed_query("test")

    def test_embed_batch_raises_on_connection_error(
        self,
        provider: OllamaEmbeddingProvider,
        mock_ollama_client: MagicMock,
    ) -> None:
        provider._available = True  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        mock_ollama_client.embed.side_effect = ollama.ResponseError("timeout")

        with pytest.raises(EmbeddingError, match="Embedding failed"):
            provider.embed_batch(["text"])


# -- EmbeddingService (factory) --


class TestEmbeddingServiceFactory:
    """EmbeddingService caches providers per model."""

    @patch("ragnest.services.embedding_service.ollama.Client")
    def test_get_provider_returns_provider(
        self, mock_client_cls: MagicMock,  # noqa: ARG002
    ) -> None:
        service = EmbeddingService("http://localhost:11434")

        provider = service.get_provider("bge-m3")

        assert isinstance(provider, OllamaEmbeddingProvider)

    @patch("ragnest.services.embedding_service.ollama.Client")
    def test_get_provider_caches(
        self, mock_client_cls: MagicMock,  # noqa: ARG002
    ) -> None:
        service = EmbeddingService("http://localhost:11434")

        p1 = service.get_provider("bge-m3")
        p2 = service.get_provider("bge-m3")

        assert p1 is p2

    @patch("ragnest.services.embedding_service.ollama.Client")
    def test_get_provider_different_models(
        self, mock_client_cls: MagicMock,  # noqa: ARG002
    ) -> None:
        service = EmbeddingService("http://localhost:11434")

        p1 = service.get_provider("bge-m3")
        p2 = service.get_provider("gte-qwen2")

        assert p1 is not p2

    @patch("ragnest.services.embedding_service.ollama.Client")
    def test_list_models_returns_names(
        self, mock_client_cls: MagicMock,
    ) -> None:
        mock_client = mock_client_cls.return_value
        mock_client.list.return_value = SimpleNamespace(
            models=[
                SimpleNamespace(model="bge-m3:latest"),
                SimpleNamespace(model="gte-qwen2:7b"),
            ],
        )

        service = EmbeddingService("http://localhost:11434")
        models = service.list_models()

        assert models == ["bge-m3:latest", "gte-qwen2:7b"]

    @patch("ragnest.services.embedding_service.ollama.Client")
    def test_list_models_returns_empty_on_error(
        self, mock_client_cls: MagicMock,
    ) -> None:
        mock_client = mock_client_cls.return_value
        mock_client.list.side_effect = ollama.ResponseError("unavailable")

        service = EmbeddingService("http://localhost:11434")
        models = service.list_models()

        assert models == []
