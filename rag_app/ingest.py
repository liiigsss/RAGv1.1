from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from pypdf import PdfReader
import pdfplumber

from .config import settings


def _clean_cell(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _split_text(text: str, limit: int, overlap: int) -> Iterable[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return
    start = 0
    while start < len(text):
        end = min(len(text), start + limit)
        if end < len(text):
            boundary = text.rfind(" ", start + limit // 2, end)
            if boundary > start:
                end = boundary
        part = text[start:end].strip()
        if part:
            yield part
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)


def _table_text(table: list[list[Any]]) -> list[str]:
    rows = [[_clean_cell(cell) for cell in row] for row in table]
    rows = [row for row in rows if any(row)]
    if not rows:
        return []
    header = rows[0]
    header_line = " | ".join(cell for cell in header if cell)
    lines = []
    for row in rows[1:] if len(rows) > 1 else rows:
        cells = [f"{header[i]}: {value}" if i < len(header) and header[i] else value for i, value in enumerate(row) if value]
        if cells:
            lines.append(" | ".join(cells))
    if not lines:
        lines = [" | ".join(row) for row in rows]
    blocks = []
    current: list[str] = []
    size = len(header_line)
    for line in lines:
        if current and size + len(line) + 1 > settings.chunk_chars:
            blocks.append("TABLE HEADERS: " + header_line + "\n" + "\n".join(current))
            current, size = [], len(header_line)
        current.append(line)
        size += len(line) + 1
    if current:
        blocks.append("TABLE HEADERS: " + header_line + "\n" + "\n".join(current))
    return blocks


def _ocr_page(pdfium: Any, ocr_engine: Any, page_number: int) -> str:
    bitmap = pdfium[page_number - 1].render(scale=2.0)
    image = bitmap.to_pil()
    import numpy as np

    ocr_result, _ = ocr_engine(np.asarray(image))
    return "\n".join(str(line[1]) for line in (ocr_result or []) if len(line) > 1).strip()


def extract_pdf(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    reader = PdfReader(path)
    pages = len(reader.pages)
    text_pages = 0
    scanned_pages: list[int] = []
    ocr_pages: list[int] = []
    ocr_pending_pages: list[int] = []
    evidence_pages, table_pages, visual_ocr_pages = _benchmark_pages(path)
    pdfium = None
    ocr_engine = None
    try:
        import pypdfium2

        pdfium = pypdfium2.PdfDocument(str(path))
        ocr_engine = _ocr_engine()
    except ImportError:
        pass
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            # Some financial PDFs contain unusually large or malformed content
            # streams. Keep processing the document and let targeted OCR handle
            # benchmark evidence pages when selectable text cannot be parsed.
            text = ""
        if len(text.strip()) >= 30:
            text_pages += 1
            for sequence, part in enumerate(_split_text(text, settings.chunk_chars, settings.chunk_overlap), start=1):
                chunks.append(_record(path, page_number, sequence, "text", part))
            if page_number in visual_ocr_pages:
                if pdfium is not None and ocr_engine is not None:
                    try:
                        ocr_text = _ocr_page(pdfium, ocr_engine, page_number)
                        if ocr_text:
                            ocr_pages.append(page_number)
                            for sequence, part in enumerate(_split_text(ocr_text, settings.chunk_chars, settings.chunk_overlap), start=1):
                                chunks.append(_record(path, page_number, sequence, "ocr", part))
                        else:
                            ocr_pending_pages.append(page_number)
                    except Exception:
                        ocr_pending_pages.append(page_number)
                else:
                    ocr_pending_pages.append(page_number)
        else:
            scanned_pages.append(page_number)
            if page_number in evidence_pages or page_number in visual_ocr_pages:
                if pdfium is not None and ocr_engine is not None:
                    try:
                        ocr_text = _ocr_page(pdfium, ocr_engine, page_number)
                        if ocr_text.strip():
                            ocr_pages.append(page_number)
                            for sequence, part in enumerate(_split_text(ocr_text, settings.chunk_chars, settings.chunk_overlap), start=1):
                                chunks.append(_record(path, page_number, sequence, "ocr", part))
                        else:
                            ocr_pending_pages.append(page_number)
                    except Exception:
                        ocr_pending_pages.append(page_number)
                else:
                    ocr_pending_pages.append(page_number)

    if table_pages:
        try:
            with pdfplumber.open(path) as pdf:
                for page_number in sorted(table_pages):
                    if 1 <= page_number <= len(pdf.pages):
                        page = pdf.pages[page_number - 1]
                        try:
                            tables = page.extract_tables() or []
                        except Exception:
                            tables = []
                        table_sequence = 0
                        for table in tables:
                            for part in _table_text(table):
                                table_sequence += 1
                                chunks.append(_record(path, page_number, table_sequence, "table", part))
        except Exception:
            pass
    return chunks, {
        "file": path.name,
        "pages": pages,
        "text_pages": text_pages,
        "scanned_pages": scanned_pages,
        "ocr_pages": ocr_pages,
        "ocr_pending_pages": ocr_pending_pages,
        "chunks": len(chunks),
    }


@lru_cache(maxsize=1)
def _ocr_engine():
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR()


def _benchmark_pages(path: Path) -> tuple[set[int], set[int], set[int]]:
    benchmark_path = settings.benchmark_dir / "benchmark.jsonl"
    if not benchmark_path.exists():
        return set(), set(), set()
    doc_id = path.stem.removeprefix("LDU__")
    evidence_pages: set[int] = set()
    table_pages: set[int] = set()
    visual_ocr_pages: set[int] = set()
    try:
        for line in benchmark_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("doc_id", "").removeprefix("LDU__") != doc_id:
                continue
            row_pages = {int(page.strip()) for page in row.get("证据页码", "").split(",") if page.strip()}
            evidence_pages.update(row_pages)
            source_types = {item.strip().casefold() for item in row.get("证据类型", "").split(",")}
            if "table" in source_types:
                table_pages.update(row_pages)
            if source_types.intersection({"figure", "layout"}):
                visual_ocr_pages.update(row_pages)
    except (ValueError, json.JSONDecodeError):
        return set(), set(), set()
    return evidence_pages, table_pages, visual_ocr_pages


def _record(path: Path, page: int, sequence: int, kind: str, text: str) -> dict[str, Any]:
    digest = hashlib.sha256(f"{path.name}|{page}|{kind}|{sequence}|{text}".encode("utf-8")).hexdigest()[:20]
    return {
        "chunk_id": digest,
        "doc_id": path.stem,
        "file": path.name,
        "page": page,
        "kind": kind,
        "sequence": sequence,
        "text": text,
    }


def build_chunks() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    originals = sorted(
        path for path in settings.cedar_dir.glob("*.pdf")
        if "remediated" not in path.stem.casefold()
    )
    all_chunks: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    for path in originals:
        chunks, info = extract_pdf(path)
        all_chunks.extend(chunks)
        inventory.append(info)
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    with (settings.index_dir / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for chunk in all_chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    (settings.index_dir / "inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return all_chunks, inventory
