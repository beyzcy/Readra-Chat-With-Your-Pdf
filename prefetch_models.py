"""Download every model once, at image build time, so the first visitor does not wait.

Used by the Dockerfile. Safe to run locally too: python prefetch_models.py
"""

from fastembed import TextEmbedding

from rag import EMBED_MODEL, RERANKER_MODEL

print("embeddings:", EMBED_MODEL)
TextEmbedding(model_name=EMBED_MODEL)

if RERANKER_MODEL.lower() not in ("", "off", "none", "0"):
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    print("reranker:", RERANKER_MODEL)
    TextCrossEncoder(model_name=RERANKER_MODEL)

print("ocr")
from rapidocr import RapidOCR  # noqa: E402

RapidOCR()
print("models ready")
