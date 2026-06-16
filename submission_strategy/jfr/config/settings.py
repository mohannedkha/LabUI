"""Application-level paths and runtime settings."""

import os
from pathlib import Path
from functools import lru_cache
from pydantic import BaseModel


class Settings(BaseModel):
    data_dir: Path = Path.home() / ".local" / "share" / "jfr"
    db_path: Path = Path.home() / ".local" / "share" / "jfr" / "db.sqlite"
    vectors_dir: Path = Path.home() / ".local" / "share" / "jfr" / "vectors"
    attachments_dir: Path = Path.home() / ".local" / "share" / "jfr" / "attachments"
    models_dir: Path = Path.home() / ".local" / "share" / "jfr" / "models"
    journals_yaml: Path = Path(__file__).parent.parent.parent / "data" / "journals.yaml"
    policy_toml: Path = Path(__file__).parent.parent.parent / "data" / "policy.toml"

    # Embedding models.
    # specter2_base = plain encoder (no PEFT adapters) — 768-dim scientific text.
    # specter2 (with LoRA adapters) requires a matching peft version; use base for now.
    abstract_model: str = "allenai/specter2_base"
    claim_model: str = "BAAI/bge-large-en-v1.5"

    # Corpus
    corpus_window_months: int = 36
    top_k_neighbors: int = 25

    # Web
    web_host: str = "127.0.0.1"
    web_port: int = 8765

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.vectors_dir, self.attachments_dir, self.models_dir):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    # Allow env overrides. JFR_DATA_DIR relocates the whole data tree; all the
    # sub-paths (db, vectors, attachments, models) are derived from it so a single
    # env var gives a fully independent instance (e.g. LabUI keeps its own data).
    if override := os.environ.get("JFR_DATA_DIR"):
        base = Path(override).expanduser()
        s = s.model_copy(update={
            "data_dir":        base,
            "db_path":         base / "db.sqlite",
            "vectors_dir":     base / "vectors",
            "attachments_dir": base / "attachments",
            "models_dir":      base / "models",
        })
    if host := os.environ.get("JFR_WEB_HOST"):
        s = s.model_copy(update={"web_host": host})
    if port := os.environ.get("JFR_WEB_PORT"):
        s = s.model_copy(update={"web_port": int(port)})
    return s
