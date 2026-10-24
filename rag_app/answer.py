from __future__ import annotations

import json
import os
import re
import time
from decimal import Decimal, InvalidOperation
from typing import Any

from .config import settings


GENERATION_PROMPT_VERSION = "1.1"
JUDGE_PROMPT_VERSION = "1.1-zh"


def _answer_text(value: Any) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value or "").strip()


def _normal_format(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text.casefold() in {"none", "null", "n/a", ""} else text


def _api_key() -> str:
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("缺少 DEEPSEEK_API_KEY。请在本机 .env 中配置新 key。")
    return api_key


def _completion(payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError("请先安装 requirements.txt。") from exc

    headers = {"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"}
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            response = httpx.post(
                f"{settings.deepseek_base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            last_error = exc
            status = exc.response.status_code
            if status not in {408, 425, 429} and status < 500:
                raise
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            last_error = exc
        if attempt < 4:
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"DeepSeek 请求重试后仍失败：{type(last_error).__name__ if last_error else 'unknown'}") from last_error


def answer_strategy(question: str, profile: dict[str, Any] | None = None) -> str:
    profile = profile or {}
    question_type = str(profile.get("question_type") or "").strip().casefold()
    answer_format = _normal_format(profile.get("answer_format")).casefold()
    task_tag = str(profile.get("task_tag") or "").strip()
    if answer_format == "list" and question_type == "topic2title":
        return "从证据原文抽取标题，输出标题数组。"
    if answer_format in {"integer", "float", "number", "decimal"}:
        return "简短数值答案，保留币种、单位、期间和正负号。"
    if answer_format == "string":
        return "参考任务标签直接回答，尽量简短。"
    if answer_format == "list":
        return "按证据条目抽取并输出清单。"
    query = str(question or "").casefold()
    if re.search(r"\b(title|heading|section name|figure caption|caption)\b", query):
        return (
            "按问题范围从证据中逐字提取标题或完整图注；保留原文拼写、编号和符号，"
            "每项单独列出，不改写标题，也不添加相邻但无关的项目。"
        )
    if re.search(r"\b(list|enumerate|name all|identify all|what are the|which .* (?:were|are|was|is))\b", query):
        return "按问题限定范围逐项回答；每个被询问的对象单独列出，并保留相关数值、单位、期间和正负号，不添加无关条目。"
    if re.search(r"^\s*(does|do|did|is|are|was|were|has|have|can|could|will|would|should)\b", query):
        return "先直接回答是或否，再用一句话说明证据；注意题目中的否定措辞，不能反转判断。"
    if re.search(r"\b(how many|how much|what percentage|what percent|what amount|what value|what was the total|what is the total|calculate|how large)\b", query):
        return "简洁回答所求数值；从表格标题和列名继承适用单位、数量级、币种、期间和正负号。单位写全，不使用 b、m、k 等缩写；除非问题要求，否则不要自行换算或四舍五入。"
    if re.search(r"\b(topic|topics|risk categories|components|major risks|items|years)\b", query):
        return "按问题限定范围逐项回答；每项单独列出，保留原文名称和必要数值，不添加无关内容。"
    if task_tag:
        return "参考任务标签直接回答，尽量简短。"
    return "给出能完整回答问题的最短答案；准确保留原文中的单位、币种、期间、正负号、章节编号和特殊符号。证据不足时明确说明。"


def generate_answer(
    question: str,
    evidence: list[dict[str, Any]],
    profile: dict[str, Any] | None = None,
) -> tuple[str, str]:
    context = "\n\n".join(
        f"[{item['file']}, p.{item['page']}, chunk {item['chunk_id']}]\n{item['text']}"
        for item in evidence
    )
    profile = profile or {}
    guidance = answer_strategy(question, profile)
    system_prompt = (
        "Answer using only the supplied document evidence. Do not use outside knowledge. "
        "Be concise and preserve financial values, dates, currency, units, signs, and reporting periods exactly. "
        "For table values, bind each number to its row label, column period, and applicable table units. "
        "Spell scale units such as billion and thousand in full. Do not silently round, rescale, or reverse signs. "
        "Use task_tag, question_type, and answer_format only as guidance for answer shape; they do not supply answer facts. "
        "Do not use the gold answer or reference pages when generating. For lists, answer only the requested scope "
        "and put each requested item on its own line. "
        "If evidence is insufficient, return a concise abstention and no citations. "
        "Cite only pages whose supplied passages support the answer. Return a JSON object with keys "
        "answer and citations; citations must be an array of objects with file and page."
    )
    payload = {
        "model": settings.generation_model,
        "thinking": {"type": "disabled"},
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Answer guidance from task metadata and question: {guidance}\n"
                    f"task_tag: {profile.get('task_tag') or ''}; "
                    f"question_type: {profile.get('question_type') or ''}; "
                    f"answer_format: {_normal_format(profile.get('answer_format')) or 'unspecified'}\n\n"
                    f"Question: {question}\n\nEvidence:\n{context}"
                ),
            },
        ],
        "response_format": {"type": "json_object"},
    }
    data = _completion(payload, timeout=180.0)
    content = data["choices"][0]["message"]["content"]
    result = json.loads(content)
    answer = _answer_text(result.get("answer", ""))
    citations = result.get("citations", [])
    if isinstance(citations, str):
        try:
            citations = json.loads(citations)
        except json.JSONDecodeError:
            citations = []
    if not isinstance(citations, list):
        citations = []
    normalized_citations = []
    for citation in citations:
        if not isinstance(citation, dict):
            continue
        try:
            normalized_citations.append({"file": str(citation["file"]), "page": int(citation["page"])})
        except (KeyError, TypeError, ValueError):
            continue
    return answer, json.dumps(normalized_citations, ensure_ascii=False)


def _citation_list(citations_json: str) -> list[dict[str, Any]]:
    try:
        citations = json.loads(citations_json)
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(citations, list):
        return []
    result = []
    for item in citations:
        if isinstance(item, dict):
            try:
                result.append({"file": str(item["file"]), "page": int(item["page"])})
            except (KeyError, TypeError, ValueError):
                pass
    return result


def judge_answer(
    question: str,
    gold_answer: str,
    generated_answer: str,
    profile: dict[str, Any],
    citations_json: str,
    evidence: list[dict[str, Any]],
    reference_pages: str = "",
) -> dict[str, Any]:
    citations = _citation_list(citations_json)
    cited_blocks = []
    for citation in citations:
        passages = [
            item["text"] for item in evidence
            if item.get("file") == citation["file"] and int(item.get("page", -1)) == citation["page"]
        ]
        cited_blocks.append({
            "file": citation["file"],
            "page": citation["page"],
            "retrieved_passages": passages,
        })

    answer_format = _normal_format(profile.get("answer_format"))
    list_applicable = answer_format.casefold() == "list"
    format_applicable = bool(answer_format)
    system_prompt = (
        "你是严谨的检索问答评审员。请分别评估生成答案的语义、清单条目、格式、引用支持和标准答案是否与原文冲突。"
        "语义允许等义表达，但实体、关键事实、金额、数量、单位、币种、正负号和期间必须正确。"
        "如果标准答案标注为无法回答，而生成答案给出了实质性答案，即使检索证据支持该答案，语义仍判为错误；"
        "若所给原文证据明确显示标准答案的无法回答标注有误，应另将标准答案冲突标为是。"
        "清单题分别判断必需条目是否齐全、是否出现无关条目；标题抽取题允许采用证据中逐字出现且直接回答问题的其他标题，"
        "不强求可能不完整的标准标题清单，但应扣除无关标题。"
        "引用支持只判断引用页面的所给证据是否支持生成答案，不得用标准答案代替证据判断。"
        "只要引用页面内容支持答案，就不要求页码与基准标注页完全相同；没有提供引用页证据的引用一律不受支持。"
        "所有理由字段必须只用简体中文书写，不得夹带英文句子、英文标签或英文缩写；尽量写明具体缺失事实、数值差异、"
        "引用页支持了什么或标准答案与原文哪里冲突，不要只写笼统的对错结论。金额、单位、期间和页码须准确保留。"
        "只返回 JSON，布尔判断字段为 semantic_correct、list_items_correct、format_correct、citation_supported、benchmark_conflict；"
        "理由字段为 semantic_reason、list_reason、format_reason、citation_reason、benchmark_conflict_reason。"
        "非清单题的 list_items_correct 返回 null；未指定答案格式时 format_correct 返回 null。"
    )
    payload = {
        "model": settings.generation_model,
        "thinking": {"type": "disabled"},
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps({
                    "question": question,
                    "task_tag": profile.get("task_tag"),
                    "question_type": profile.get("question_type"),
                    "answer_format": answer_format or "unspecified",
                    "gold_answer": gold_answer,
                    "generated_answer": generated_answer,
                    "citations": citations,
                    "benchmark_reference_pages_for_review_only": reference_pages,
                    "cited_page_passages": cited_blocks,
                    "all_retrieved_evidence_for_answer_and_benchmark_review": [
                        {"file": item.get("file"), "page": item.get("page"), "text": item.get("text", "")}
                        for item in evidence
                    ],
                    "has_required_list_score": list_applicable,
                    "has_required_format_score": format_applicable,
                }, ensure_ascii=False),
            },
        ],
        "response_format": {"type": "json_object"},
    }
    data = _completion(payload, timeout=180.0)
    result = json.loads(data["choices"][0]["message"]["content"])
    return {
        "semantic": int(bool(result.get("semantic_correct"))),
        "semantic_reason": str(result.get("semantic_reason", "")),
        "list_items": (int(bool(result.get("list_items_correct"))) if list_applicable else "N/A"),
        "list_reason": str(result.get("list_reason", "")),
        "format": (int(bool(result.get("format_correct"))) if format_applicable else "N/A"),
        "format_reason": str(result.get("format_reason", "")),
        "citation_supported": int(bool(result.get("citation_supported"))),
        "citation_reason": str(result.get("citation_reason", "")),
        "benchmark_conflict": int(bool(result.get("benchmark_conflict"))),
        "benchmark_conflict_reason": str(result.get("benchmark_conflict_reason", "")),
    }


def _currency_name(value: str | None) -> str:
    text = (value or "").strip().upper()
    aliases = {
        "$": "USD", "US$": "USD", "A$": "AUD", "B$": "BND", "C$": "CAD",
        "€": "EUR", "£": "GBP", "¥": "CNY", "₩": "KRW", "RMB": "CNY",
    }
    return aliases.get(text, text)


def _numeric_facts(text: Any) -> list[tuple[Decimal, str, str, int]]:
    if isinstance(text, (list, dict)):
        text = json.dumps(text, ensure_ascii=False)
    normalized = str(text or "").replace("−", "-").replace("–", "-")
    negative_parentheses = re.compile(
        r"(?P<currency>US\$|A\$|B\$|C\$|USD|AUD|BND|CAD|EUR|GBP|CNY|RMB|JPY|KRW|[$€£¥₩])?\s*"
        r"\(\s*(?P<number>\d{1,3}(?:,\d{3})+|\d+)(?:\.(?P<decimal>\d+))?\s*\)"
    )
    normalized = negative_parentheses.sub(
        lambda match: f"{match.group('currency') or ''} -{match.group('number')}"
        + (f".{match.group('decimal')}" if match.group("decimal") else ""),
        normalized,
    )
    pattern = re.compile(
        r"(?<![A-Za-z0-9])(?P<currency>US\$|A\$|B\$|C\$|USD|AUD|BND|CAD|EUR|GBP|CNY|RMB|JPY|KRW|[$€£¥₩])?\s*"
        r"(?P<number>[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*"
        r"(?P<unit>%|percent(?:age)?(?:\s+points?)?|bps?|basis\s+points?|trillion|billion|million|thousand|tn|bn|mn|b|k|m)?"
        r"(?![A-Za-z])",
        re.IGNORECASE,
    )
    scales = {
        "trillion": Decimal("1e12"), "tn": Decimal("1e12"),
        "billion": Decimal("1e9"), "bn": Decimal("1e9"), "b": Decimal("1e9"),
        "million": Decimal("1e6"), "mn": Decimal("1e6"), "m": Decimal("1e6"),
        "thousand": Decimal("1e3"), "k": Decimal("1e3"),
    }
    found: list[tuple[Decimal, str, str, int]] = []
    for match in pattern.finditer(normalized):
        raw = match.group("number").replace(",", "")
        try:
            value = Decimal(raw)
        except InvalidOperation:
            continue
        raw_unit = (match.group("unit") or "").casefold().strip()
        scale = scales.get(raw_unit, Decimal(1))
        if raw_unit in scales:
            value *= scale
            unit = ""
        elif raw_unit in {"percent", "percentage", "%", "percent point", "percentage point", "percent points", "percentage points"}:
            unit = "%"
        elif raw_unit in {"bps", "bp", "basis point", "basis points"}:
            value /= Decimal(100)
            unit = "%"
        else:
            unit = raw_unit
        decimal_places = max(0, -Decimal(raw).as_tuple().exponent) - scale.adjusted() if scale != 1 else max(0, -Decimal(raw).as_tuple().exponent)
        found.append((value.normalize(), _currency_name(match.group("currency")), unit, decimal_places))
    return found


def _is_unanswerable(text: str) -> bool:
    normalized = re.sub(r"[^a-z]+", " ", str(text or "").casefold()).strip()
    return normalized in {
        "not answerable", "unanswerable", "not enough information", "cannot be answered",
        "cannot answer", "insufficient information", "n a", "none",
    }


def numeric_check(gold: str, answer: str, answer_format: Any = None) -> tuple[int | str, str]:
    fmt = _normal_format(answer_format).casefold()
    if fmt == "list":
        return "N/A", "清单题单独评估各项，不将标题中的数字当作数值事实。"
    expected = _numeric_facts(gold)
    if _is_unanswerable(gold) and not expected:
        return "N/A", "标准答案为不可回答；数值项不适用。"
    if not expected:
        if fmt in {"integer", "float", "number", "numeric"}:
            return 0, "数值格式题的标准答案中未能解析出数值，需人工复核基准答案。"
        return "N/A", "标准答案没有可核对的数值事实；数值项不适用。"
    actual = _numeric_facts(answer)
    if not actual:
        return 0, "标准答案含数值，但生成答案未解析到数值。"

    remaining = list(actual)
    for expected_value, expected_currency, expected_unit, precision in expected:
        tolerance = Decimal("0.5") * (Decimal(10) ** (-precision))
        matched = None
        for index, (actual_value, actual_currency, actual_unit, _) in enumerate(remaining):
            currency_matches = not expected_currency or actual_currency == expected_currency
            unit_matches = not expected_unit or actual_unit == expected_unit
            if currency_matches and unit_matches and abs(actual_value - expected_value) <= tolerance:
                matched = index
                break
        if matched is None:
            currency_labels = {
                "USD": "美元", "AUD": "澳元", "BND": "文莱元", "CAD": "加元",
                "EUR": "欧元", "GBP": "英镑", "CNY": "人民币", "JPY": "日元",
            }
            expected_currency_label = currency_labels.get(expected_currency, expected_currency)
            expected_value_display = format(expected_value, "f")
            return 0, (
                f"缺少或不匹配的数值、单位、币种或正负号："
                f"{expected_value_display} {expected_currency_label} {expected_unit}"
            ).strip()
        remaining.pop(matched)
    return 1, "标准答案中的数值事实均匹配；已处理千分位、科学计数法和常见金额单位。"
