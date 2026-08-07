# supercompress_local

A standalone, **dependency-free** prompt/context compressor, ported from your
`Supercompress-main` repo's local browser/CLI engine
(`packages/proxy/src/assets/compress-engine.js`) into pure Python.

- **No `pip install` needed.** Only the Python standard library (`re`, `json`,
  `math`, `os`). Copy the `supercompress_local/` folder into any project and
  import it.
- **Uses your actual trained model.** `model.json` (the same weights your JS
  engine ships) is bundled and loaded to score tokens/lines.
- **Works with plain Python, LangChain, or LangGraph** — the core has zero
  framework dependency; the two adapter files only import their frameworks
  lazily, inside the functions that need them.

## What's ported

Faithfully ported from `compress-engine.js`:
- Content routing + the three domain preprocessors (JSON SmartCrusher, Code
  compressor, Log/Trace compressor)
- Line classification, question-entity extraction, text normalization
- Token-level feature extraction (16-dim) + the trained neural scorer
  (same `linear → layernorm → gelu` stack, same weights)
- Duplicate-line penalty, query-relevance scoring, block segmentation/scoring

Simplified vs. the original `selectCompilerLines`: the JS engine has ~1,200
lines of tuning specific to public QA benchmarks (HotpotQA-style multi-hop
passage cover-sets, TREC few-shot type-bank detection, hand-built
role/synonym dictionaries). That's benchmark-fitting, not general-purpose
reuse, so it's replaced here with a general greedy knapsack-style block
selector that keeps high-value blocks under a token budget, preserves
structural dependencies (headings before definitions, imports before
definitions, log lines around stack traces), and closes any markdown fences
a cut would otherwise leave open.

For very large inputs (> `MAX_MODEL_TOKENS`, default 6000 tokens) the neural
forward pass — which is pure-Python, so O(tokens) with no vectorization —
is skipped in favor of the cheap deterministic attention score already
computed per token. Block-level entity/keyword/IDF scoring still runs in
full either way. Raise `MAX_MODEL_TOKENS` in `compressor.py` if you don't
mind slower calls on huge contexts, or lower it if latency matters more.

## Quick start (plain Python)

```python
from supercompress_local import SuperCompress

sc = SuperCompress()
result = sc.compress(
    context="...your long context / retrieved docs / chat history...",
    query="What broke in the payments service?",
    budget_ratio=0.35,   # keep ~35% of tokens
)

print(result.compressed_text)
print(f"{result.original_tokens} -> {result.kept_tokens} tokens "
      f"({result.kv_savings_pct:.1f}% saved)")
```

`SuperCompress()` loads `model.json` once and can be reused across calls —
create it once at startup, not per-request.

## LangChain

```python
from supercompress_local.langchain_adapter import (
    make_compressing_runnable, compress_documents, SuperCompressRetriever,
)

# 1) Drop into an LCEL chain as a Runnable
compressor = make_compressing_runnable(query=user_query, budget_ratio=0.3)
chain = retriever | format_docs | compressor | prompt | llm

# 2) Compress a list of retrieved Documents directly
docs = retriever.invoke(user_query)
docs = compress_documents(docs, user_query, budget_ratio=0.3)

# 3) Wrap a retriever so every call is compressed automatically
compressing_retriever = SuperCompressRetriever(retriever, budget_ratio=0.3)
docs = compressing_retriever.invoke(user_query)
```

## LangGraph

```python
from supercompress_local.langgraph_adapter import make_compression_node

graph.add_node("compress", make_compression_node(budget_ratio=0.3))
graph.add_edge("compress", "call_model")
```

Or call it directly on a message list (list of `{"role", "content"}` dicts,
or LangChain `BaseMessage` objects — both work):

```python
from supercompress_local.langgraph_adapter import compress_messages

result = compress_messages(state["messages"], budget_ratio=0.3)
state["messages"] = result["messages"]
```

This mirrors `assembleMessages` from your `proxy/src/compressor.js`: the last
user message becomes the untouched `query`, everything before it gets
compressed into one system-role summary message, and any leading system
prompt is kept as-is.

## Files

```
supercompress_local/
  __init__.py            # exports SuperCompress, CompressResult, compress, load_model
  compressor.py           # the whole engine — pure stdlib, no dependencies
  model.json               # bundled trained weights (same as your JS engine)
  langchain_adapter.py     # optional, imports langchain_core lazily
  langgraph_adapter.py     # optional, works with dicts or LangChain messages
```

## Notes

- `budget_ratio` is a *target*, not a hard ceiling — blocks that clearly match
  your query (error traces, matching definitions, keyword hits) can push
  actual retention a bit above the target rather than get cut.
- `result.line_annotations` gives a per-line kept/dropped trace with a reason
  string, handy for debugging what got cut.
- If `model.json` fails to load for any reason, the compressor still works —
  it falls back to the deterministic (non-neural) attention heuristic.
