"""Configuration for Ragnest — Pydantic Settings with YAML loader."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings

from ragnest.models.domain import KBConfig


class DBSettings(BaseModel):
    """Database connection settings."""

    backend: Literal["postgres", "supabase"] = "postgres"
    host: str = "localhost"
    port: int = Field(default=5432, ge=1, le=65535)
    name: str = "rag_hub"
    user: str = "postgres"
    password: SecretStr = SecretStr("postgres")

    @property
    def connection_string(self) -> str:
        return (
            f"postgresql://{self.user}:{self.password.get_secret_value()}"
            f"@{self.host}:{self.port}/{self.name}"
        )


class OllamaSettings(BaseModel):
    """Ollama embedding service settings."""

    base_url: str = "http://localhost:11434"


class DefaultsSettings(BaseModel):
    """Default chunking settings applied when not overridden per KB."""

    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    separator: str = "\n\n"


class StateSettings(BaseModel):
    """SQLite state database settings."""

    path: str = "~/.ragnest/state.db"


class AppSettings(BaseSettings):
    """Top-level application settings with env var override support."""

    databases: dict[str, DBSettings] = Field(
        default_factory=lambda: {"default": DBSettings()}
    )
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    defaults: DefaultsSettings = Field(default_factory=DefaultsSettings)
    knowledge_bases: dict[str, KBConfig] = Field(default_factory=dict)
    state: StateSettings = Field(default_factory=StateSettings)

    model_config = {"env_prefix": "RAGNEST_", "env_nested_delimiter": "__"}

    @property
    def database(self) -> DBSettings:
        """Backward-compatible access to the first (default) backend settings."""
        return next(iter(self.databases.values()))


def load_settings(config_path: str | None = None) -> AppSettings:
    """Load settings from YAML file with env var overrides."""
    if config_path is None:
        config_path = os.environ.get(
            "RAGNEST_CONFIG",
            str(Path(__file__).parent.parent.parent / "config.yaml"),
        )

    path = Path(config_path)
    if not path.exists():
        return AppSettings()

    with path.open() as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    # Load .env file if it exists (secrets live here, not in config.yaml)
    env_path = path.parent / ".env"
    if env_path.exists():
        from dotenv import load_dotenv  # noqa: PLC0415

        load_dotenv(env_path)

    defaults_raw = raw.get("defaults", {})
    ollama_raw = raw.get("ollama", {})
    state_raw = raw.get("state", {})

    kbs: dict[str, KBConfig] = {}
    for kb_name, kb_raw in raw.get("knowledge_bases", {}).items():
        kbs[kb_name] = KBConfig(
            name=kb_name,
            description=kb_raw.get("description", ""),
            model=kb_raw.get("model", "bge-m3"),
            dimensions=kb_raw.get("dimensions", 1024),
            chunk_size=kb_raw.get(
                "chunk_size", defaults_raw.get("chunk_size", 1000)
            ),
            chunk_overlap=kb_raw.get(
                "chunk_overlap", defaults_raw.get("chunk_overlap", 200)
            ),
            separator=kb_raw.get(
                "separator", defaults_raw.get("separator", "\n\n")
            ),
        )

    # Build databases dict: support both singular and plural config keys
    databases: dict[str, DBSettings] = {}
    if "databases" in raw:
        # Advanced: named backends
        for db_name, db_raw in raw["databases"].items():
            databases[db_name] = DBSettings(**db_raw)
    else:
        # Simple: single backend wrapped as "default"
        db_raw = raw.get("database", {})
        # Env vars override YAML for secrets (RAGNEST_DATABASE__PASSWORD, etc.)
        for key in ("host", "port", "name", "user", "password"):
            env_val = os.environ.get(f"RAGNEST_DATABASE__{key.upper()}")
            if env_val is not None:
                db_raw[key] = int(env_val) if key == "port" else env_val
        databases["default"] = DBSettings(**db_raw)

    return AppSettings(
        databases=databases,
        ollama=OllamaSettings(**ollama_raw) if ollama_raw else OllamaSettings(),
        defaults=DefaultsSettings(**defaults_raw) if defaults_raw else DefaultsSettings(),
        knowledge_bases=kbs,
        state=StateSettings(**state_raw) if state_raw else StateSettings(),
    )


# Backward compatibility aliases for existing code during migration
Config = AppSettings
DBConfig = DBSettings


def load_config(config_path: str | None = None) -> AppSettings:
    """Backward-compatible alias for load_settings."""
    return load_settings(config_path)
