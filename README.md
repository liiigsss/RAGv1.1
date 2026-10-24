# Cedar RAG 基线

本项目用 Cedar 原始 PDF 建立本地混合检索 RAG。`*_remediated.pdf` 仅作为后续对照素材，不会进入基线索引。

## 当前数据核对结果

- 测试基准是 `pdfs - Google Sheets.pdf`，LongDocURL 题目通过 question ID 与 `文件/source/LongDocURL_public.jsonl` 对齐，保留基准 PDF 的出题顺序和原始文本列。
- LongDocURL 原始 JSONL 提供证据页、答案类型等评估元数据；canonical 问题与答案用于避免 PDF 导出换行造成的单词断裂。
- FinanceBench 的 3 条题目会保留在基准中。当前 Cedar 目录没有它们对应的 Nike、PayPal、Walmart 原始 PDF，评测时会标为不可运行，不会混入 LongDocURL 的准确率分母。
- 首轮模型已按用户确认配置：Qwen3-Embedding-0.6B、Qwen3-Reranker-0.6B；生成模型使用 DeepSeek API 的 `deepseek-flash`。

## 安装

建议使用 Python 3.11 或 3.12 的独立虚拟环境。当前 `.venv` 已安装 CUDA 版 PyTorch，并验证可调用 RTX 3060 Laptop GPU。已有文档向量索引先前由 CPU 建成；本次检索查询向量和 reranker 使用 CUDA。无需仅为改变计算设备而重建索引。

在本机 `.env` 中配置 DeepSeek API key；不要把密钥写入工作簿或提交到版本库。聊天中曾公开过的 key 应在服务商处撤销。

本机模型为 `Qwen/Qwen3-Embedding-0.6B` 与 `Qwen/Qwen3-Reranker-0.6B`。OCR 对基准引用到的扫描证据页运行；其他扫描页会在清单中标记。Embedding 与 reranker 按适配 6 GB 显存的批量设置运行。若驱动或模型加载报显存不足，可在 `.env` 中将 `RAG_DEVICE=cpu`。索引生成会在 `data/index/embeddings.partial.npy` 和进度文件中保存断点；评测检查点为 `outputs/evaluation_v1.1_checkpoint.jsonl`，中断后再次运行 `evaluate` 可从已完成题目恢复。

## 使用

```powershell
python -m rag_app.cli benchmark-import
python -m rag_app.cli ingest
python -m rag_app.cli index
python -m rag_app.cli evaluate
```

`benchmark-import` 会创建 `data/benchmark/benchmark_master.xlsx`。`evaluate` 会按数据集生成独立 Excel；本轮文件为 `outputs/LongDocURL_accuracy_v1.1.xlsx` 和 `outputs/FinanceBench_accuracy_v1.1.xlsx`。LongDocURL 工作簿含 `准确率汇总`、`测评结果`、`检索明细`、`生成证据` 和 `评分说明` 五个工作表；候选检索预览、各路排名与分数、实际发送给模型的完整证据、引用支持判断及复核标记均保存在 Excel 中。评分单元格可编辑，严格通过列由公式计算。

混合检索候选池为 24，reranker 重排后取 Top-K 8，并可加入最多 8 个同页相邻片段。语义评分允许等义表达；数值评分核对关键数值、单位、币种、正负号和期间；引用页不必与基准页码完全相同，但引用文本必须支持答案。缺少原始 PDF 的题目会保留在工作簿中，不计入准确率分母。

## 运行范围

目前是面向单机验证的本地工作台。索引、检索和结果文件保存在本机。经本轮授权，题目、检索证据、标准答案和生成答案会发送到 DeepSeek API 用于生成与评分；API 密钥保存在本机 `.env`，不写入 Excel。对敏感金融资料上线前，应再补充访问控制、日志脱敏、密钥托管、备份和部署审计。
