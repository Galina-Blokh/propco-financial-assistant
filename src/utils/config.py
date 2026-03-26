from pathlib import Path

from dotenv import load_dotenv
from pydantic import ConfigDict
from pydantic_settings import BaseSettings

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API Keys — required, no defaults
    openai_api_key: str

    # LLM backend: "openai" (default) or "local" (llama-cpp-python)
    llm_backend: str = "openai"

    # Model config
    openai_model_reasoning: str = "gpt-4o-mini"
    openai_model_extraction: str = "gpt-4o-mini"
    openai_temperature_extraction: float = 0.0
    openai_temperature_reasoning: float = 0.2

    # Local GGUF model settings (used when llm_backend=local)
    local_model_path: str = ""
    local_n_ctx: int = 2048
    local_n_threads: int = 4
    local_n_gpu_layers: int = 0

    # Paths
    cortex_data_path: str = str(Path(__file__).resolve().parents[2] / "cortex.parquet")

    # MLflow
    mlflow_tracking_uri: str = "sqlite:///mlruns.db"
    mlflow_experiment_name: str = "real_estate_agent"

    # Application
    log_level: str = "INFO"
    enable_cache: bool = True
    cache_ttl_seconds: int = 3600
    max_tool_calls: int = 3
    max_turns: int = 5

    # KV Cache
    kv_cache_enabled: bool = True
    kv_cache_max_entries: int = 50
    kv_cache_ttl_seconds: int = 3600
    kv_cache_backend: str = "memory"


settings = Settings()
