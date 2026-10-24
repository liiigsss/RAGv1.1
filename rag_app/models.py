from __future__ import annotations

from typing import Sequence

import numpy as np

from .config import settings


def _device(torch):
    if settings.device != "auto":
        return settings.device
    return "cuda" if torch.cuda.is_available() else "cpu"


class QwenEmbedder:
    def __init__(self, model_name: str | None = None):
        try:
            import torch
            import torch.nn.functional as functional
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("请先安装 requirements.txt，并按 README 安装适配显卡的 PyTorch。") from exc
        self.torch = torch
        self.functional = functional
        self.device = _device(torch)
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name or settings.embedding_model, padding_side="left")
        self.model = AutoModel.from_pretrained(model_name or settings.embedding_model, torch_dtype=dtype).to(self.device).eval()

    def encode(self, texts: Sequence[str], is_query: bool = False, batch_size: int = 4) -> np.ndarray:
        instruction = "Given a financial and long-document question, retrieve passages that contain the evidence needed to answer it accurately."
        inputs_text = [f"Instruct: {instruction}\nQuery:{text}" for text in texts] if is_query else list(texts)
        result = []
        for start in range(0, len(inputs_text), batch_size):
            batch = self.tokenizer(
                inputs_text[start:start + batch_size], padding=True, truncation=True,
                max_length=2048, return_tensors="pt"
            ).to(self.device)
            with self.torch.inference_mode():
                output = self.model(**batch).last_hidden_state
                vectors = output[:, -1]
                vectors = self.functional.normalize(vectors, p=2, dim=1)
            result.append(vectors.float().cpu().numpy())
        if not result:
            return np.empty((0, 0), dtype=np.float32)
        return np.concatenate(result, axis=0)


class QwenReranker:
    def __init__(self, model_name: str | None = None):
        try:
            import torch
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError("请先安装 requirements.txt，并按 README 安装适配显卡的 PyTorch。") from exc
        self.device = _device(torch)
        kwargs = {"torch_dtype": torch.float16} if self.device.startswith("cuda") else {}
        self.model = CrossEncoder(
            model_name or settings.reranker_model,
            device=self.device,
            max_length=1024,
            model_kwargs=kwargs,
            prompts={"finance_retrieval": "Given a financial and long-document question, judge whether the document passage contains evidence that answers it."},
            default_prompt_name="finance_retrieval",
        )

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        pairs = [(query, passage) for passage in passages]
        values = self.model.predict(pairs, batch_size=4, show_progress_bar=False)
        return [float(value) for value in values]
