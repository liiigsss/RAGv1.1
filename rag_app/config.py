from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "source"
if not SOURCE_DIR.exists() and (ROOT / "文件" / "source").exists():
    SOURCE_DIR = ROOT / "文件" / "source"


@dataclass(frozen=True)
class Settings:
    cedar_dir: Path = ROOT / "Cedar"
    source_dir: Path = SOURCE_DIR
    index_dir: Path = Path(os.getenv("RAG_INDEX_DIR", str(ROOT / "data" / "index")))
    benchmark_dir: Path = ROOT / "data" / "benchmark"
    output_dir: Path = ROOT / "outputs"
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    reranker_model: str = os.getenv("RERANKER_MODEL", "Qwen/Qwen3-Reranker-0.6B")
    generation_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    chunk_chars: int = int(os.getenv("CHUNK_CHARS", "1800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "240"))
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "8"))
    candidate_k: int = int(os.getenv("CANDIDATE_K", "24"))
    device: str = os.getenv("RAG_DEVICE", "auto")


settings = Settings()
