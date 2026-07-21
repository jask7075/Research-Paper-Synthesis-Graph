"""Single source of runtime configuration.

Layering (lowest → highest precedence):
    configs/settings.yaml  →  environment variables (RPSG_*)  →  .env

Usage:
    from rpsg.config import get_settings
    settings = get_settings()
    settings.models.judge_model  # "claude-opus-4-8"
"""

from __future__ import annotations

import functools
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SETTINGS_YAML = PROJECT_ROOT / "configs" / "settings.yaml"


class Paths(BaseModel):
    data_raw: Path = Path("data/raw")
    data_interim: Path = Path("data/interim")
    data_processed: Path = Path("data/processed")
    data_external: Path = Path("data/external")
    eval_gold: Path = Path("eval/gold")
    eval_runs: Path = Path("eval/runs")
    kuzu_db: Path = Path("data/processed/rpsg.kuzu")
    vector_index: Path = Path("data/processed/vectors.faiss")

    def resolved(self) -> "Paths":
        """Return a copy with every path made absolute against the project root."""
        return Paths(**{k: (PROJECT_ROOT / v) for k, v in self.model_dump().items()})


class Models(BaseModel):
    extraction_model: str = "claude-haiku-4-5"
    judge_model: str = "claude-opus-4-8"
    synthesis_model: str = "claude-opus-4-8"
    local_inference_model: str = "Qwen/Qwen2.5-14B-Instruct-AWQ"


class Embeddings(BaseModel):
    model_name: str = "allenai/specter2_base"
    dim: int = 768
    batch_size: int = 32


class Chunking(BaseModel):
    target_tokens: int = 512
    overlap_tokens: int = 64
    respect_sections: bool = True


class Calibration(BaseModel):
    min_quadratic_kappa: float = 0.6
    length_bias_alpha: float = 0.05


class Eval(BaseModel):
    judge_temperature: float = 0.0
    calibration: Calibration = Field(default_factory=Calibration)


class Settings(BaseSettings):
    """Root settings object. Secrets are read from the environment / .env."""

    model_config = SettingsConfigDict(
        env_prefix="RPSG_",
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Secrets (from .env / env; not in settings.yaml)
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    s2_api_key: str | None = Field(default=None, alias="S2_API_KEY")
    grobid_url: str = "http://localhost:8070"
    pg_dsn: str | None = None

    # Structured config (defaults overridden by settings.yaml via load())
    paths: Paths = Field(default_factory=Paths)
    models: Models = Field(default_factory=Models)
    embeddings: Embeddings = Field(default_factory=Embeddings)
    chunking: Chunking = Field(default_factory=Chunking)
    eval: Eval = Field(default_factory=Eval)


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once (yaml defaults, then env/.env overrides) and cache."""
    raw = _load_yaml(_SETTINGS_YAML)
    # `ingestion` block is consumed directly by the S2 client; keep it addressable.
    settings = Settings(**raw)
    settings.paths = settings.paths.resolved()
    return settings