from __future__ import annotations

import argparse

from .benchmark import build_benchmark
from .config import settings
from .ingest import build_chunks
from .pipeline import ask, build_index, run_benchmark
from .reports import export_evaluation_reports



def main() -> None:
    parser = argparse.ArgumentParser(description="Cedar 企业 RAG 基线工具")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("benchmark-import", help="按测试 PDF 行序构建标准题库及 Excel")
    commands.add_parser("ingest", help="提取 Cedar 原始 PDF 文本和表格")
    commands.add_parser("index", help="提取文档并生成 Qwen 向量索引")
    evaluate_parser = commands.add_parser("evaluate", help="运行混合检索、重排、DeepSeek 生成并保存逐题诊断数据")
    evaluate_parser.add_argument("--question-ids", nargs="+", help="只运行指定 question_id；省略则运行完整基准")
    evaluate_parser.add_argument("--rerun-completed", action="store_true", help="强制重跑所选题目，即使检查点中已有结果")
    ask_parser = commands.add_parser("ask", help="对本地索引提问")
    ask_parser.add_argument("question")
    args = parser.parse_args()

    if args.command == "benchmark-import":
        jsonl_path, xlsx_path, rows = build_benchmark()
        print(f"基准题目：{len(rows)} 条")
        print(f"JSONL：{jsonl_path}")
        print(f"Excel：{xlsx_path}")
    elif args.command == "ingest":
        chunks, inventory = build_chunks()
        print(f"原始 PDF：{len(inventory)} 份；文本片段：{len(chunks)} 个")
        print(f"清单：{settings.index_dir / 'inventory.json'}")
    elif args.command == "index":
        summary = build_index()
        print(f"已建立索引：{summary}")
    elif args.command == "evaluate":
        selected_ids = set(args.question_ids) if args.question_ids else None
        checkpoints = run_benchmark(question_ids=selected_ids, rerun_completed=args.rerun_completed)
        for path in checkpoints:
            print(f"Checkpoint: {path}")
            for report_path in export_evaluation_reports(path):
                print(f"Evaluation report: {report_path}")
    elif args.command == "ask":
        answer, evidence = ask(args.question)
        print(answer)
        for item in evidence:
            print(f"- {item['file']} p.{item['page']} ({item.get('reranker_score', item.get('rrf_score')):.3f})")


if __name__ == "__main__":
    main()
