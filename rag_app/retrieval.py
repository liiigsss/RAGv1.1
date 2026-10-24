from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from .config import settings
from .models import QwenEmbedder, QwenReranker


def _tokens(text: str) -> list[str]:
    text = text.casefold()
    # Keep numbers, English words, and individual CJK characters searchable.
    return re.findall(r"[a-z0-9]+(?:[.,%/-][a-z0-9]+)*|[\u3400-\u9fff]", text)


class HybridRetriever:
    def __init__(self, use_reranker: bool = True):
        if not (settings.index_dir / "chunks.jsonl").exists() or not (settings.index_dir / "embeddings.npy").exists():
            raise FileNotFoundError("索引还不存在，请先运行 `python -m rag_app.cli index`。")
        self.chunks = [json.loads(line) for line in (settings.index_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
        self.embeddings = np.load(settings.index_dir / "embeddings.npy")
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RuntimeError("请先安装 requirements.txt。") from exc
        self.bm25 = BM25Okapi([_tokens(chunk["text"]) for chunk in self.chunks])
        self.embedder = QwenEmbedder()
        self.reranker = QwenReranker() if use_reranker else None

    def search_with_trace(
        self, query: str, top_k: int | None = None, candidate_k: int | None = None
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        top_k = top_k or settings.retrieval_k
        candidate_k = candidate_k or settings.candidate_k
        query_vector = self.embedder.encode([query], is_query=True)[0]
        dense_scores = self.embeddings @ query_vector
        dense_order = np.argsort(-dense_scores)[:candidate_k]
        lexical_scores = self.bm25.get_scores(_tokens(query))
        lexical_order = np.argsort(-lexical_scores)[:candidate_k]
        dense_ranks = {int(index): rank for rank, index in enumerate(dense_order, start=1)}
        lexical_ranks = {int(index): rank for rank, index in enumerate(lexical_order, start=1)}
        ranks: dict[int, float] = {}
        for order in (dense_order, lexical_order):
            for rank, index in enumerate(order, start=1):
                ranks[int(index)] = ranks.get(int(index), 0.0) + 1.0 / (60 + rank)
        candidate_ids = sorted(ranks, key=ranks.get, reverse=True)[:candidate_k]
        results = []
        for candidate_rank, idx in enumerate(candidate_ids, start=1):
            item = dict(self.chunks[idx])
            item["dense_score"] = float(dense_scores[idx])
            item["bm25_score"] = float(lexical_scores[idx])
            item["rrf_score"] = ranks[idx]
            item["dense_rank"] = dense_ranks.get(idx)
            item["bm25_rank"] = lexical_ranks.get(idx)
            item["candidate_rank"] = candidate_rank
            item["reranker_score"] = None
            results.append(item)
        if self.reranker and results:
            rerank_scores = self.reranker.score(query, [r["text"] for r in results])
            for item, score in zip(results, rerank_scores):
                item["reranker_score"] = score
            results.sort(key=lambda r: r["reranker_score"], reverse=True)
        for rank, item in enumerate(results, start=1):
            item["final_rank"] = rank
            item["selected_for_generation"] = rank <= top_k
        return results[:top_k], results

    def search(self, query: str, top_k: int | None = None, candidate_k: int | None = None) -> list[dict[str, Any]]:
        selected, _ = self.search_with_trace(query, top_k=top_k, candidate_k=candidate_k)
        return selected
