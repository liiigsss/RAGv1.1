from __future__ import annotations

import json
import hashlib
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from .answer import (
    GENERATION_PROMPT_VERSION,
    JUDGE_PROMPT_VERSION,
    answer_strategy,
    generate_answer,
    judge_answer,
    numeric_check,
)
from .benchmark import load_benchmark
from .config import ROOT, settings
from .ingest import build_chunks
from .models import QwenEmbedder
from .retrieval import HybridRetriever


def build_index() -> dict[str, int]:
    chunks_path = settings.index_dir / "chunks.jsonl"
    inventory_path = settings.index_dir / "inventory.json"
    if chunks_path.exists() and inventory_path.exists():
        chunks = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    else:
        chunks, inventory = build_chunks()
    if not chunks:
        raise RuntimeError("Cedar 原始 PDF 没有提取出可检索文本。")
    embedder = QwenEmbedder()
    vectors_path = settings.index_dir / "embeddings.npy"
    partial_path = settings.index_dir / "embeddings.partial.npy"
    progress_path = settings.index_dir / "embeddings.progress.json"
    fingerprint = hashlib.sha256(
        (settings.embedding_model + "\n" + "\n".join(chunk["chunk_id"] for chunk in chunks)).encode("utf-8")
    ).hexdigest()
    progress: dict[str, Any] = {}
    if progress_path.exists() and partial_path.exists():
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            progress = {}
    resumable = (
        progress.get("fingerprint") == fingerprint
        and progress.get("chunk_count") == len(chunks)
        and partial_path.exists()
    )
    if not resumable:
        partial_path.unlink(missing_ok=True)
        progress = {"fingerprint": fingerprint, "chunk_count": len(chunks), "completed_indices": []}
        progress_path.write_text(json.dumps(progress), encoding="utf-8")
    completed_indices = set(int(index) for index in progress.get("completed_indices", [])) if resumable else set()

    old_vectors = None
    old_chunk_map: dict[str, int] = {}
    old_index_dir = ROOT / "data" / "index"
    if settings.index_dir.resolve() != old_index_dir.resolve():
        old_manifest_path = old_index_dir / "manifest.json"
        old_chunks_path = old_index_dir / "chunks.jsonl"
        old_vectors_path = old_index_dir / "embeddings.npy"
        if old_manifest_path.exists() and old_chunks_path.exists() and old_vectors_path.exists():
            try:
                old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))
                if old_manifest.get("embedding_model") == settings.embedding_model:
                    old_chunks = [
                        json.loads(line) for line in old_chunks_path.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    ]
                    old_vectors = np.load(old_vectors_path, mmap_mode="r")
                    if len(old_chunks) == old_vectors.shape[0]:
                        old_chunk_map = {chunk["chunk_id"]: index for index, chunk in enumerate(old_chunks)}
                    else:
                        old_vectors = None
                        old_chunk_map = {}
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                old_vectors = None
                old_chunk_map = {}

    vector_shape = None
    if resumable:
        vectors = np.lib.format.open_memmap(partial_path, mode="r+")
        if vectors.shape[0] != len(chunks):
            vectors._mmap.close()
            partial_path.unlink(missing_ok=True)
            progress = {"fingerprint": fingerprint, "chunk_count": len(chunks), "completed_indices": []}
            progress_path.write_text(json.dumps(progress), encoding="utf-8")
            return build_index()
        vector_shape = vectors.shape
    else:
        first_new = next((i for i, chunk in enumerate(chunks) if chunk["chunk_id"] not in old_chunk_map), None)
        if first_new is None:
            vector_shape = (len(chunks), int(old_vectors.shape[1]))
        else:
            first = embedder.encode([chunks[first_new]["text"]], is_query=False, batch_size=1)
            vector_shape = (len(chunks), int(first.shape[1]))
            vectors = np.lib.format.open_memmap(partial_path, mode="w+", dtype=np.float32, shape=vector_shape)
            vectors[first_new] = first[0]
            completed_indices.add(first_new)
            progress["completed_indices"] = sorted(completed_indices)
            progress_path.write_text(json.dumps(progress), encoding="utf-8")
        if first_new is None:
            vectors = np.lib.format.open_memmap(partial_path, mode="w+", dtype=np.float32, shape=vector_shape)

    reused_indices = set()
    if old_vectors is not None:
        for index, chunk in enumerate(chunks):
            old_index = old_chunk_map.get(chunk["chunk_id"])
            if old_index is not None and old_vectors.shape[1] == vectors.shape[1]:
                vectors[index] = old_vectors[old_index]
                reused_indices.add(index)

    batch_size = 4
    pending_indices = [
        index for index in range(len(chunks))
        if index not in reused_indices and index not in completed_indices
    ]
    for start in range(0, len(pending_indices), batch_size):
        batch_indices = pending_indices[start:start + batch_size]
        encoded = embedder.encode(
            [chunks[index]["text"] for index in batch_indices], is_query=False, batch_size=batch_size
        )
        vectors[batch_indices] = encoded
        vectors.flush()
        completed_indices.update(batch_indices)
        progress["completed_indices"] = sorted(completed_indices)
        progress_path.write_text(json.dumps(progress), encoding="utf-8")
        total_done = len(reused_indices) + len(completed_indices)
        if total_done % 80 < batch_size or total_done == len(chunks):
            print(f"向量化进度：{total_done}/{len(chunks)}（复用 {len(reused_indices)}，新增 {len(completed_indices)}）")
    if len(reused_indices) + len(completed_indices) != len(chunks):
        vectors._mmap.close()
        raise RuntimeError("索引向量仍有未完成的片段，暂不发布索引。")
    vectors.flush()
    dimension = int(vectors.shape[1])
    vectors._mmap.close()
    del vectors
    if old_vectors is not None:
        old_vectors._mmap.close()
    os.replace(partial_path, vectors_path)
    progress_path.unlink(missing_ok=True)
    manifest = {
        "embedding_model": settings.embedding_model,
        "chunk_count": len(chunks),
        "dimension": dimension,
        "document_count": len(inventory),
        "reused_chunk_count": len(reused_indices),
        "newly_encoded_chunk_count": len(completed_indices),
    }
    (settings.index_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"documents": len(inventory), "chunks": len(chunks), "dimension": dimension}


def _citation_scores(
    citations_json: str, evidence: list[dict[str, Any]], source_row: dict[str, Any]
) -> tuple[int, int | str, list[dict[str, Any]], str]:
    try:
        citations = json.loads(citations_json)
    except (TypeError, json.JSONDecodeError):
        citations = []
    if not isinstance(citations, list):
        citations = []
    normalized = []
    for citation in citations:
        if isinstance(citation, dict):
            try:
                normalized.append({"file": str(citation["file"]), "page": int(citation["page"])})
            except (KeyError, TypeError, ValueError):
                continue
    retrieved = {(str(item["file"]), int(item["page"])) for item in evidence}
    citation_hit = int(bool(normalized) and all((item["file"], item["page"]) in retrieved for item in normalized))
    reference_pages = {
        int(page.strip()) for page in str(source_row.get("证据页码") or "").split(",")
        if page.strip().isdigit()
    }
    if not reference_pages:
        reference_match: int | str = "N/A"
    else:
        original_file = str(source_row.get("对应原始PDF") or "")
        reference_match = int(any(
            item["file"] == original_file and item["page"] in reference_pages for item in normalized
        ))
    if not normalized:
        note = "答案未提供可用引用。"
    elif not citation_hit:
        note = "至少一个引用页不在实际发送给生成模型的上下文中。"
    elif reference_match == 0:
        note = "引用页未命中基准标注页；此字段仅作对照，最终依据引用内容判定。"
    else:
        note = "引用页均出现在实际生成上下文中。"
    return citation_hit, reference_match, normalized, note


def _expand_neighbor_context(
    selected: list[dict[str, Any]], chunks: list[dict[str, Any]], neighbor_budget: int = 8
) -> list[dict[str, Any]]:
    selected_ids = {item["chunk_id"] for item in selected}
    lookup: dict[tuple[str, int, str, int], dict[str, Any]] = {}
    for chunk in chunks:
        try:
            key = (str(chunk["file"]), int(chunk["page"]), str(chunk["kind"]), int(chunk["sequence"]))
        except (KeyError, TypeError, ValueError):
            continue
        lookup[key] = chunk

    expanded: list[dict[str, Any]] = []
    added: set[str] = set()
    neighbors_added = 0
    for rank, item in enumerate(selected, start=1):
        primary = dict(item)
        primary["evidence_role"] = "primary"
        primary["prompt_order"] = len(expanded) + 1
        primary["primary_rank"] = rank
        if primary["chunk_id"] not in added:
            expanded.append(primary)
            added.add(primary["chunk_id"])

        key_base = (str(item["file"]), int(item["page"]), str(item["kind"]))
        adjacent = [lookup.get((*key_base, int(item["sequence"]) + delta)) for delta in (-1, 1)]
        adjacent = [chunk for chunk in adjacent if chunk is not None]
        for neighbor in adjacent:
            if neighbors_added >= neighbor_budget:
                break
            if neighbor["chunk_id"] in selected_ids or neighbor["chunk_id"] in added:
                continue
            contextual = dict(neighbor)
            contextual["evidence_role"] = "neighbor_context"
            contextual["prompt_order"] = len(expanded) + 1
            contextual["primary_rank"] = None
            contextual["neighbor_for_chunk_id"] = item["chunk_id"]
            expanded.append(contextual)
            added.add(contextual["chunk_id"])
            neighbors_added += 1
    for item in expanded:
        item["text"] = str(item.get("text") or "")
    return expanded


def _append_checkpoint(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def _chinese_reason(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    text = text.replace("N/A", "不适用").replace("List", "清单").replace("list", "清单")
    if not text or re.search(r"[A-Za-z]", text):
        return fallback
    return text


def run_benchmark(
    question_ids: set[str] | None = None, rerun_completed: bool = False
) -> list[Path]:
    rows = load_benchmark()
    retriever = HybridRetriever(use_reranker=True)
    files = {path.name for path in settings.cedar_dir.glob("*.pdf") if "remediated" not in path.stem.casefold()}
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    run_version = os.getenv("RAG_EVALUATION_VERSION", "v1.1")
    checkpoint_path = Path(os.getenv(
        "RAG_EVALUATION_CHECKPOINT",
        str(settings.output_dir / f"evaluation_{run_version}_checkpoint.jsonl"),
    ))
    benchmark_overrides: dict[int, str] = {}
    if run_version == "v1.2":
        override_path = settings.benchmark_dir / "v1.2_benchmark_review.json"
        if override_path.exists():
            override_data = json.loads(override_path.read_text(encoding="utf-8"))
            benchmark_overrides = {
                int(seq): str(item["说明"])
                for seq, item in override_data.items()
                if isinstance(item, dict) and item.get("说明")
            }
    completed: dict[str, dict[str, Any]] = {}
    if checkpoint_path.exists():
        for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                saved = json.loads(line)
                if saved.get("运行状态") in {"完成", "缺少原始PDF"}:
                    completed[saved["question_id"]] = saved
    selected_rows = [row for row in rows if question_ids is None or row["question_id"] in question_ids]
    if question_ids is not None:
        found = {row["question_id"] for row in selected_rows}
        missing_ids = question_ids - found
        if missing_ids:
            raise ValueError(f"以下 question_id 不在基准中：{', '.join(sorted(missing_ids))}")
    for index, source_row in enumerate(selected_rows, start=1):
        question_id = source_row["question_id"]
        if question_id in completed and not rerun_completed:
            print(f"[{index}/{len(selected_rows)}] restored checkpoint: {question_id}")
            continue
        row = dict(source_row)
        row["检索候选明细"] = []
        row["生成证据明细"] = []
        row["生成模型"] = settings.generation_model
        row["生成提示版本"] = GENERATION_PROMPT_VERSION
        row["评分提示版本"] = JUDGE_PROMPT_VERSION
        row["Embedding模型"] = settings.embedding_model
        row["Reranker模型"] = settings.reranker_model
        row["向量设备"] = retriever.embedder.device
        row["候选池"] = settings.candidate_k
        row["最终TopK"] = settings.retrieval_k
        row["答案策略"] = ""
        if not row.get("对应原始PDF") or row["对应原始PDF"] not in files:
            row["复核备注"] = "跳过生成：基准引用的原始 PDF 不在 Cedar。"
            row["运行状态"] = "缺少原始PDF"
            _append_checkpoint(checkpoint_path, row)
            print(f"[{index}/{len(selected_rows)}] skip missing document: {row['question_id']}")
            continue
        profile = {
            "task_tag": row.get("task_tag"),
            "question_type": row.get("question_type"),
            "answer_format": row.get("answer_format"),
        }
        row["答案策略"] = answer_strategy(row["问题"], profile)

        selected, candidates = retriever.search_with_trace(
            row["问题"], top_k=settings.retrieval_k, candidate_k=settings.candidate_k
        )
        selected_ids = {item["chunk_id"] for item in selected}
        evidence = _expand_neighbor_context(selected, retriever.chunks)
        evidence_by_id = {item["chunk_id"]: item for item in evidence}
        row["检索候选明细"] = []
        for item in candidates:
            trace = dict(item)
            trace["evidence_role"] = (
                "primary" if item["chunk_id"] in selected_ids
                else "neighbor_context" if item["chunk_id"] in evidence_by_id
                else "candidate"
            )
            trace["included_for_generation"] = item["chunk_id"] in evidence_by_id
            trace["prompt_order"] = evidence_by_id.get(item["chunk_id"], {}).get("prompt_order")
            row["检索候选明细"].append(trace)
        candidate_ids = {item["chunk_id"] for item in candidates}
        for item in evidence:
            if item["chunk_id"] not in candidate_ids:
                trace = dict(item)
                trace.update({
                    "dense_score": None, "bm25_score": None, "rrf_score": None,
                    "dense_rank": None, "bm25_rank": None, "candidate_rank": None,
                    "reranker_score": None, "final_rank": None,
                    "selected_for_generation": False, "included_for_generation": True,
                })
                row["检索候选明细"].append(trace)
        row["生成证据明细"] = evidence

        answer, citations_json = generate_answer(row["问题"], evidence, profile=profile)
        citation_hit, citation_reference_match, citations, citation_trace_note = _citation_scores(
            citations_json, evidence, source_row
        )
        judged = judge_answer(
            question=row["问题"],
            gold_answer=row["标准答案"],
            generated_answer=answer,
            profile=profile,
            citations_json=citations_json,
            evidence=evidence,
            reference_pages=str(row.get("证据页码") or ""),
        )
        manual_conflict = benchmark_overrides.get(int(row["序号"]))
        if manual_conflict:
            judged["benchmark_conflict"] = 1
            judged["benchmark_conflict_reason"] = manual_conflict
        numeric, numeric_note = numeric_check(
            row["标准答案"], answer, answer_format=row.get("answer_format")
        )
        semantic_fallback = (
            "生成答案与标准答案含义一致。" if judged["semantic"] == 1
            else "生成答案与标准答案在关键事实或回答范围上存在差异，建议结合证据复核。"
        )
        semantic_reason = _chinese_reason(judged.get("semantic_reason"), semantic_fallback)
        numeric_note = _chinese_reason(numeric_note, "数值核对理由未能生成中文，请人工复核。")
        list_fallback = (
            "清单中的必需条目和相关性均符合要求。" if judged["list_items"] == 1
            else "清单可能缺少必需条目或包含无关条目，建议逐项对照标准答案和证据。"
        )
        list_reason = (
            "本题不要求清单回答。" if str(judged["list_items"]).upper() == "N/A"
            else _chinese_reason(judged.get("list_reason"), list_fallback)
        )
        format_fallback = (
            "生成答案符合指定格式。" if judged["format"] == 1
            else "生成答案未完全遵守指定格式。"
        )
        format_reason = (
            "基准未指定答案格式。" if str(judged["format"]).upper() == "N/A"
            else _chinese_reason(judged.get("format_reason"), format_fallback)
        )
        citation_reason = _chinese_reason(
            judged.get("citation_reason"),
            "引用证据支持答案中的关键内容。" if judged["citation_supported"] else "引用证据未能支持答案中的关键内容。",
        )
        citation_trace_note = _chinese_reason(
            citation_trace_note,
            "引用页面未能与实际发送给模型的证据对应，请人工复核。",
        )
        conflict_reason = _chinese_reason(
            judged.get("benchmark_conflict_reason"),
            "标准答案与原文证据存在明显冲突，建议人工核对。"
            if judged["benchmark_conflict"] else "未发现标准答案与原文证据存在明确冲突。",
        )
        citation_supported = int(bool(citation_hit and judged["citation_supported"]))
        row["生成答案"] = answer
        row["检索引用"] = json.dumps(citations, ensure_ascii=False)
        row["语义正确(0/1)"] = judged["semantic"]
        row["数值正确(0/1/N/A)"] = numeric
        row["List条目匹配(0/1/N/A)"] = judged["list_items"]
        row["格式符合(0/1/N/A)"] = judged["format"]
        row["基准答案需复核(0/1)"] = judged["benchmark_conflict"]
        row["基准冲突说明"] = conflict_reason
        row["引用命中生成上下文(0/1)"] = citation_hit
        row["引用命中基准证据页(0/1/N/A)"] = citation_reference_match
        row["引用内容支持(0/1)"] = citation_supported
        row["评分理由"] = "；".join([
            f"语义核对：{semantic_reason}",
            f"数值核对：{numeric_note}",
            f"清单核对：{list_reason}",
            f"格式核对：{format_reason}",
            f"引用内容：{citation_reason}",
            f"引用链路：{citation_trace_note}",
            f"基准冲突：{conflict_reason}",
        ])
        review_items = []
        if judged["semantic"] != 1:
            review_items.append(f"语义核对未通过：{semantic_reason}")
        if numeric not in {1, "N/A"}:
            review_items.append(f"数值核对未通过：{numeric_note}")
        if judged["list_items"] == 0:
            review_items.append(f"清单核对未通过：{list_reason}")
        if judged["format"] == 0:
            review_items.append(f"格式核对未通过：{format_reason}")
        if citation_hit != 1:
            review_items.append(f"引用链路未通过：{citation_trace_note}")
        if citation_supported != 1:
            review_items.append(f"引用内容未通过：{citation_reason}")
        if judged["benchmark_conflict"] == 1:
            review_items.append(f"标准答案需复核：{conflict_reason}")
        row["复核备注"] = "；".join(review_items) if review_items else "本题各项核对均通过。"
        row["运行状态"] = "完成"
        _append_checkpoint(checkpoint_path, row)
        print(f"[{index}/{len(selected_rows)}] completed: {row['question_id']}")
    return [checkpoint_path]


def ask(question: str) -> tuple[str, list[dict[str, Any]]]:
    retriever = HybridRetriever(use_reranker=True)
    evidence = retriever.search(question)
    answer, _ = generate_answer(question, evidence)
    return answer, evidence
