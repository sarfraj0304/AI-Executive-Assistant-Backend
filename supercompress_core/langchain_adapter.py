"""
Optional LangChain glue for supercompress_local. LangChain itself is NOT a
dependency of this package — this file only imports it lazily, inside the
functions below, so `import supercompress_local` never requires LangChain to
be installed. If you don't use LangChain, ignore this file entirely.
"""

from __future__ import annotations

from typing import Any, Optional

from .compressor import SuperCompress


def make_compressing_runnable(query: str, budget_ratio: float = 0.35, sc: Optional[SuperCompress] = None):
    """
    Build a LangChain Runnable that compresses a plain string (e.g. a
    retrieved-docs blob or long chat history rendered to text).

        from supercompress_local.langchain_adapter import make_compressing_runnable

        compressor = make_compressing_runnable(query="What changed in v2?", budget_ratio=0.3)
        chain = retriever | format_docs | compressor | prompt | llm
    """
    from langchain_core.runnables import RunnableLambda  # lazy import

    engine = sc or SuperCompress()

    def _run(text: str) -> str:
        return engine.compress(text, query=query, budget_ratio=budget_ratio).compressed_text

    return RunnableLambda(_run)


def compress_documents(documents: list, query: str, budget_ratio: float = 0.35, sc: Optional[SuperCompress] = None) -> list:
    """
    Compress a list of LangChain `Document` objects in place (returns new
    Documents with `.page_content` replaced by the compressed text).

        from supercompress_local.langchain_adapter import compress_documents
        docs = retriever.invoke(query)
        docs = compress_documents(docs, query, budget_ratio=0.3)
    """
    from langchain_core.documents import Document  # lazy import

    engine = sc or SuperCompress()
    out = []
    for doc in documents:
        result = engine.compress(doc.page_content, query=query, budget_ratio=budget_ratio)
        meta = dict(doc.metadata or {})
        meta["supercompress"] = {
            "original_tokens": result.original_tokens,
            "kept_tokens": result.kept_tokens,
            "kv_savings_pct": round(result.kv_savings_pct, 1),
        }
        out.append(Document(page_content=result.compressed_text, metadata=meta))
    return out


class SuperCompressRetriever:
    """
    Wraps any LangChain retriever (or object with `.invoke`/`.get_relevant_documents`)
    and compresses the returned documents' text against the query before they
    reach the LLM.

        from supercompress_local.langchain_adapter import SuperCompressRetriever
        compressing_retriever = SuperCompressRetriever(base_retriever, budget_ratio=0.3)
        docs = compressing_retriever.invoke("What changed in v2?")
    """

    def __init__(self, base_retriever: Any, budget_ratio: float = 0.35, sc: Optional[SuperCompress] = None):
        self.base_retriever = base_retriever
        self.budget_ratio = budget_ratio
        self.engine = sc or SuperCompress()

    def invoke(self, query: str, *args, **kwargs) -> list:
        if hasattr(self.base_retriever, "invoke"):
            docs = self.base_retriever.invoke(query, *args, **kwargs)
        else:
            docs = self.base_retriever.get_relevant_documents(query, *args, **kwargs)
        return compress_documents(docs, query, self.budget_ratio, self.engine)

    def get_relevant_documents(self, query: str, *args, **kwargs) -> list:
        return self.invoke(query, *args, **kwargs)


__all__ = ["make_compressing_runnable", "compress_documents", "SuperCompressRetriever"]
