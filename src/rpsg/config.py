"""Single source of runtime configuration.

Layering (lowest → highest precedence):
    configs/settings.yaml  →  environment variables (RPSG_*)  →  .env

Usage:
    from rpsg.config import get_settings
    settings = get_settings()
    settings.models.judge_model  # "gpt-5.4-mini"
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

    def resolved(self) -> Paths:
        """Return a copy with every path made absolute against the project root."""
        return Paths(**{k: (PROJECT_ROOT / v) for k, v in self.model_dump().items()})


class Models(BaseModel):
    extraction_model: str = "gpt-5.4-nano"
    judge_model: str = "gpt-5.4-mini"
    synthesis_model: str = "gpt-5.4-mini"
    local_inference_model: str = "Qwen/Qwen2.5-14B-Instruct-AWQ"
    #: Base URL of an OpenAI-compatible server, e.g. "http://localhost:8000/v1".
    #:
    #: vLLM serves the OpenAI chat-completions API, so §3.3's "route chat calls to vLLM" is
    #: exactly "point an OpenAI client at a different base URL". The plumbing lives here
    #: rather than waiting with the rest of 3.3 because it is required whichever hardware
    #: eventually serves the model, and it touches nothing 3.5 measures.
    #:
    #: Wired and unexercised: the model above needs ~8.5GB in 4-bit and CUDA-only AWQ
    #: kernels, while the development machine is an 8GB M2 where Metal caps the usable
    #: working set near 5.3GB. 3.3's run waits for hardware, not for code.
    local_inference_base_url: str | None = None
    #: Sampling temperature for the judge, and for the judge only. `None` restores the
    #: provider default, which is what Iteration 1 and 2 ran on: nothing set a temperature
    #: anywhere, so every reported kappa was computed from temperature-1.0 samples. Judging
    #: the same 34 answers three times with an identical rubric produced per-criterion kappa
    #: spreads of 0.12-0.25 -- wider than the margin by which §6 certified three criteria.
    #: A grader that cannot reproduce its own grade cannot certify anything, so this is
    #: pinned. Extraction and synthesis are deliberately left alone; changing their sampling
    #: would make Iteration 2's stored runs non-comparable for no measured benefit.
    judge_temperature: float | None = 0.0
    #: Override provider routing. None = infer from the model id (see rpsg.llm).
    provider: str | None = None
    #: Optional per-model rates, e.g. {"gpt-5.4-nano": {"input_per_mtok": 0.05,
    #: "output_per_mtok": 0.40}}. Deliberately empty by default — see rpsg.llm.usage
    #: for why rates are not hardcoded. Token counts are reported either way.
    pricing: dict[str, dict[str, float]] = Field(default_factory=dict)


class Embeddings(BaseModel):
    #: SPECTER (original), natively packaged for sentence-transformers. Chosen over
    #: `allenai/specter2_base`: SPECTER2 is an adapter model (base weights + a task
    #: adapter) and plain sentence-transformers loads only the base, silently — no
    #: error, just a model that is not the one you asked for. Same 768 dims and same
    #: scientific-paper domain, so this is a drop-in with no config knock-on.
    model_name: str = "sentence-transformers/allenai-specter"
    dim: int = 768
    batch_size: int = 32


class Extraction(BaseModel):
    #: Nodes below this confidence are dropped rather than written to the curated layer.
    #: Set to 0.65 because inspection put the real/soft boundary for `Limitation` there,
    #: and it costs only 7% of nodes.
    min_node_confidence: float = 0.65
    #: Edges are gated SEPARATELY and more loosely on purpose. A uniform 0.65 threshold
    #: retains 93% of nodes but only 71% of edges — edges are the scarce resource (350
    #: against 1379 nodes on a 20-paper corpus) and they are what makes this a graph
    #: rather than a bag of typed nodes, so paying 29% of them for node precision is a
    #: bad trade. An edge is still dropped when either endpoint was dropped, so node
    #: gating already prunes edges transitively.
    min_edge_confidence: float = 0.5
    #: Sections extracted in parallel within one paper. The stage is network-bound, not
    #: CPU-bound (measured: 0.2% CPU across a 29-minute run), so this is pure latency
    #: hiding. Raise it if the provider's rate limit allows; lower it on 429s.
    max_workers: int = 8


class Retrieval(BaseModel):
    #: Similarity is damped for chunks shorter than this (0 disables). Short text embeds
    #: near the corpus centroid and so over-scores against every query; see
    #: `rpsg.stores.vector_store._length_damping`.
    length_damping_chars: int = 800


class Chunking(BaseModel):
    target_tokens: int = 512
    overlap_tokens: int = 64
    respect_sections: bool = True


class Calibration(BaseModel):
    min_quadratic_kappa: float = 0.6
    length_bias_alpha: float = 0.05


class Eval(BaseModel):
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
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    s2_api_key: str | None = Field(default=None, alias="S2_API_KEY")
    grobid_url: str = "http://localhost:8070"
    pg_dsn: str | None = None

    # Structured config (defaults overridden by settings.yaml via load())
    paths: Paths = Field(default_factory=Paths)
    models: Models = Field(default_factory=Models)
    embeddings: Embeddings = Field(default_factory=Embeddings)
    extraction: Extraction = Field(default_factory=Extraction)
    chunking: Chunking = Field(default_factory=Chunking)
    retrieval: Retrieval = Field(default_factory=Retrieval)
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