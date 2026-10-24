from __future__ import annotations

import streamlit as st

from rag_app.config import settings
from rag_app.pipeline import ask


st.set_page_config(page_title="Cedar RAG 工作台", page_icon="📚", layout="wide")
st.title("Cedar RAG 工作台")
st.caption("当前索引仅使用 Cedar 中不带 remediated 的原始 PDF。")

inventory_path = settings.index_dir / "inventory.json"
if not inventory_path.exists():
    st.info("尚未建立索引。请先在项目目录运行 `python -m rag_app.cli index`。")
else:
    question = st.text_area("输入问题", height=110, placeholder="例如：FY2023 的经营活动现金流是多少？")
    if st.button("检索并生成答案", type="primary", disabled=not question.strip()):
        with st.spinner("正在混合检索、重排并请求 DeepSeek 生成答案…"):
            try:
                answer, evidence = ask(question.strip())
                st.subheader("答案")
                st.write(answer)
                st.subheader("引用证据")
                for item in evidence:
                    with st.expander(f"{item['file']} · 第 {item['page']} 页 · {item['kind']}"):
                        st.write(item["text"])
                        st.caption(f"融合分 {item['rrf_score']:.4f}；重排分 {item.get('reranker_score', 0):.4f}")
            except Exception as exc:
                st.error(str(exc))
