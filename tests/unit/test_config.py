"""Unit tests for configuration — settings validation, SecretStr, defaults."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from ragnest.config import (
    AppSettings,
    DBSettings,
    DefaultsSettings,
    OllamaSettings,
    StateSettings,
    load_settings,
)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


# -- DBSettings validation --


class TestDBSettingsValidation:
    """DBSettings validates port range and backend values."""

    def test_valid_settings(self) -> None:
        settings = DBSettings(
            host="db.example.com",
            port=5432,
            name="mydb",
            user="admin",
            password=SecretStr("secret"),
        )
        expected_port = 5432
        assert settings.port == expected_port
        assert settings.host == "db.example.com"

    def test_rejects_port_zero(self) -> None:
        with pytest.raises(ValidationError, match="port"):
            DBSettings(port=0)

    def test_rejects_port_too_high(self) -> None:
        with pytest.raises(ValidationError, match="port"):
            DBSettings(port=70000)

    def test_rejects_negative_port(self) -> None:
        with pytest.raises(ValidationError, match="port"):
            DBSettings(port=-1)

    def test_port_min_boundary(self) -> None:
        settings = DBSettings(port=1)
        assert settings.port == 1

    def test_port_max_boundary(self) -> None:
        settings = DBSettings(port=65535)
        expected_port = 65535
        assert settings.port == expected_port

    def test_backend_accepts_postgres(self) -> None:
        settings = DBSettings(backend="postgres")
        assert settings.backend == "postgres"

    def test_backend_accepts_supabase(self) -> None:
        settings = DBSettings(backend="supabase")
        assert settings.backend == "supabase"

    def test_backend_rejects_invalid(self) -> None:
        with pytest.raises(ValidationError, match="backend"):
            DBSettings(backend="mysql")  # type: ignore[arg-type]


# -- SecretStr protection --


class TestSecretStrProtection:
    """SecretStr must not leak password in string representations."""

    def test_password_hidden_in_str(self) -> None:
        settings = DBSettings(password=SecretStr("supersecret"))
        text = str(settings)
        assert "supersecret" not in text

    def test_password_hidden_in_repr(self) -> None:
        settings = DBSettings(password=SecretStr("supersecret"))
        text = repr(settings)
        assert "supersecret" not in text

    def test_password_accessible_via_get_secret_value(self) -> None:
        settings = DBSettings(password=SecretStr("supersecret"))
        assert settings.password.get_secret_value() == "supersecret"

    def test_connection_string_contains_actual_password(self) -> None:
        settings = DBSettings(
            host="localhost",
            port=5432,
            name="testdb",
            user="user",
            password=SecretStr("mypass"),
        )
        conn_str = settings.connection_string
        assert "mypass" in conn_str
        assert "postgresql://" in conn_str


# -- Default values --


class TestDefaultValues:
    """AppSettings applies sensible defaults when no config given."""

    def test_db_defaults(self) -> None:
        settings = DBSettings()
        expected_port = 5432
        assert settings.backend == "postgres"
        assert settings.host == "localhost"
        assert settings.port == expected_port
        assert settings.name == "rag_hub"
        assert settings.user == "postgres"

    def test_ollama_defaults(self) -> None:
        settings = OllamaSettings()
        assert settings.base_url == "http://localhost:11434"

    def test_chunk_defaults(self) -> None:
        settings = DefaultsSettings()
        expected_chunk_size = 1000
        expected_overlap = 200
        assert settings.chunk_size == expected_chunk_size
        assert settings.chunk_overlap == expected_overlap
        assert settings.separator == "\n\n"

    def test_app_settings_defaults(self) -> None:
        settings = AppSettings()
        assert settings.database.backend == "postgres"
        assert "default" in settings.databases
        assert settings.ollama.base_url == "http://localhost:11434"
        assert settings.knowledge_bases == {}

    def test_state_settings_defaults(self) -> None:
        settings = StateSettings()
        assert settings.path == "~/.ragnest/state.db"

    def test_app_settings_includes_state(self) -> None:
        settings = AppSettings()
        assert settings.state.path == "~/.ragnest/state.db"


# -- YAML config loading --


class TestLoadSettings:
    """load_settings reads YAML files and constructs AppSettings."""

    def test_load_test_config(self) -> None:
        config_path = str(FIXTURES_DIR / "config.test.yaml")
        settings = load_settings(config_path)
        assert settings.database.name == "ragnest_test"
        assert settings.database.backend == "postgres"
        expected_chunk_size = 500
        assert settings.defaults.chunk_size == expected_chunk_size

    def test_load_with_knowledge_bases(self) -> None:
        config_path = str(FIXTURES_DIR / "config.test.yaml")
        settings = load_settings(config_path)
        assert "test_kb" in settings.knowledge_bases
        kb = settings.knowledge_bases["test_kb"]
        assert kb.model == "bge-m3"
        expected_dims = 1024
        assert kb.dimensions == expected_dims

    def test_load_nonexistent_file_returns_defaults(self) -> None:
        settings = load_settings("/nonexistent/path/config.yaml")
        assert settings.database.host == "localhost"
        assert settings.knowledge_bases == {}

    def test_load_password_from_yaml(self) -> None:
        config_path = str(FIXTURES_DIR / "config.test.yaml")
        settings = load_settings(config_path)
        assert settings.database.password.get_secret_value() == "testpass"


# -- DefaultsSettings validation --


class TestDefaultsSettingsValidation:
    """DefaultsSettings validates chunk_size constraints."""

    def test_rejects_zero_chunk_size(self) -> None:
        with pytest.raises(ValidationError, match="chunk_size"):
            DefaultsSettings(chunk_size=0)

    def test_rejects_negative_chunk_size(self) -> None:
        with pytest.raises(ValidationError, match="chunk_size"):
            DefaultsSettings(chunk_size=-100)

    def test_rejects_negative_chunk_overlap(self) -> None:
        with pytest.raises(ValidationError, match="chunk_overlap"):
            DefaultsSettings(chunk_overlap=-1)
