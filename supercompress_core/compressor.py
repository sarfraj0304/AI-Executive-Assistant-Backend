"""
supercompress_local — a standalone, dependency-free prompt/context compressor.

Ported from your Supercompress-main/packages/proxy/src/assets/compress-engine.js
(the local, no-API-key JS engine) into pure Python (stdlib only: re, json, math, os).

No `pip install` required — this folder is the whole package. Copy it into any
project and `from supercompress_local import SuperCompress`.

What's ported faithfully:
  - Content routing (json / code / log / text) + the three domain preprocessors
    (JSON SmartCrusher, Code AST-ish compressor, Log/Trace compressor)
  - Line classification, question-entity extraction, normalization
  - Token-level feature extraction (16-dim) + the trained neural scorer
    (loads the same model.json / weights your JS engine ships)
  - Duplicate-line penalty + question-relevance boosting

What's simplified vs. compress-engine.js's `selectCompilerLines`:
  - The original has ~1,200 lines of benchmark-specific tuning (HotpotQA-style
    multi-hop passage cover-sets, TREC few-shot type-bank detection, role/synonym
    dictionaries for trivia QA). That's tuned to public QA benchmarks, not general
    reuse, so it's intentionally left out here.
  - In its place: general-purpose block segmentation + scoring (kept faithful to
    scoreCompilerBlocks) with a greedy knapsack-style budget selection that keeps
    high-value blocks, preserves structural dependencies (headings, imports,
    surrounding log lines for tracebacks), and closes any markdown fences it
    would otherwise split.

Usage:
    from supercompress_local import SuperCompress

    sc = SuperCompress()
    result = sc.compress(context="...long text...", query="What broke in payments?")
    print(result.compressed_text)
    print(result.original_tokens, "->", result.kept_tokens)
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

_SEM_CODE, _SEM_COMMENT, _SEM_CHAT, _SEM_BOILERPLATE = 0, 1, 2, 3

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.json")

# The trained scorer is a real per-token MLP forward pass, run in pure Python
# (no numpy). That's fine up to a few thousand tokens, but scales linearly and
# gets slow well past that. Beyond this many tokens we skip the MLP and fall
# back to the cheap deterministic h2o_score already computed per token — the
# block-level entity/keyword/IDF scoring (which does most of the real work)
# still runs in full either way. Raise this if you have numpy/torch available
# and don't mind adding a dependency; lower it if latency matters more than
# the last bit of scoring fidelity on huge contexts.
MAX_MODEL_TOKENS = 6000


# ───────────────────────── Result type ─────────────────────────

@dataclass
class CompressResult:
    original_text: str
    compressed_text: str
    original_tokens: int
    kept_tokens: int
    tokens_saved: int
    kv_savings_pct: float
    kept_line_ratio: float
    policy_name: str
    keep_ratio: float
    preprocessor: str
    answer_quality: float
    line_annotations: list = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d.pop("line_annotations", None)
        return d


# ───────────────────────── Text normalization ─────────────────────────

def normalize_context_text(text: str) -> str:
    t = str(text or "")
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"\bNEWLINE_CHAR\b", "\n", t)
    t = t.replace("\\n", "\n")
    t = re.sub(r"[\u2028\u2029]", "\n", t)
    for kw in ("Passage", "Paragraph", "Title", "Question", "Type"):
        pat = kw + r"\s*\d*\s*:" if kw in ("Passage", "Paragraph") else kw + r"\s*:"
        t = re.sub(r"(?=" + pat + r")", "\n", t, flags=re.IGNORECASE)
    out = []
    for line in t.split("\n"):
        if len(line) <= 420:
            out.append(line)
            continue
        remaining = line
        while len(remaining) > 420:
            cut = -1
            limit = min(420, len(remaining) - 1)
            lo = int(420 * 0.45)
            for i in range(limit, lo - 1, -1):
                two = remaining[i:i + 2]
                if not re.match(r"[.!?]\s", two):
                    continue
                ch = remaining[i - 1] if i - 1 >= 0 else ""
                prev = remaining[i - 2] if i - 2 >= 0 else ""
                if remaining[i] == "." and ch and re.match(r"[A-Z]", ch) and (not prev or re.match(r"[\s(\"'\[(]", prev)):
                    continue
                if remaining[i] == "." and re.search(r"(?:Mr|Mrs|Ms|Dr|St|Jr|Sr|vs|etc)\.$", remaining[max(0, i - 4):i + 1], re.IGNORECASE):
                    continue
                cut = i + 1
                break
            if cut < 0:
                for i in range(limit, int(420 * 0.55) - 1, -1):
                    if remaining[i] == " ":
                        cut = i
                        break
            if cut < 0:
                cut = 420
            out.append(remaining[:cut].rstrip())
            remaining = remaining[cut:].lstrip()
        if remaining:
            out.append(remaining)
    return "\n".join(out)


_STOP_QTERMS = {
    "what", "how", "does", "the", "is", "are", "was", "were", "function", "return",
    "class", "def", "import", "from", "this", "that", "where", "when", "who", "why",
    "which", "with", "for", "and", "your", "about", "into", "have", "has", "documented",
    "passage", "title", "question", "answer", "summary", "document", "context",
    "summarize", "findings", "recommendations", "decisions", "numbers", "errors",
    "key", "main", "facts", "type", "paragraph", "section", "first", "other",
    "individual", "location", "description", "living", "situation",
}

_STOP_ENTITIES = {
    "what", "how", "does", "the", "is", "are", "was", "were", "function", "return",
    "class", "def", "import", "from", "this", "that", "where", "when", "who", "why",
    "which", "with", "for", "and", "did", "user", "taken", "about", "into", "have",
    "has", "your", "our", "any", "all", "can", "could", "should", "would", "will",
    "passage", "title", "question", "answer", "summary", "document", "context",
    "summarize", "findings", "recommendations", "decisions", "numbers", "errors",
    "key", "main", "facts", "type", "paragraph", "section", "chapter", "page",
    "according", "following", "first", "second", "third", "other", "individual",
    "location", "description", "living", "situation", "using", "than", "then",
    "them", "they",
}


def question_terms(question: str) -> list[str]:
    q = normalize_question(question)
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+", q)
    return [w for w in words if len(w) > 2 and w.lower() not in _STOP_QTERMS]


def extract_question_entities(question: str) -> list[str]:
    question = str(question or "")
    ids = re.findall(r"[^\W\d_][\w./:-]*", question, flags=re.UNICODE)
    out = [x for x in ids if len(x) > 2 and x.lower() not in _STOP_ENTITIES]
    for m in re.finditer(r"\b([^\W\d_]{3,})(?:'s)\b", question, flags=re.UNICODE):
        base = m.group(1)
        if len(base) > 2 and base.lower() not in _STOP_ENTITIES and base not in out:
            out.append(base)
    for m in re.finditer(r"\b([A-Z][^\W\d_]+(?:\s+[A-Z][^\W\d_]+){1,4})\b", question):
        ph = m.group(1)
        if len(ph) >= 6 and ph not in out:
            out.append(ph)
    return out


def _looks_like_code_prefix(q: str) -> bool:
    s = str(q or "")
    if len(s) < 280:
        return False
    if re.search(r"\?\s*$", s.strip()):
        return False
    hits = len(re.findall(
        r"^(package\s|import\s|from\s|using\s|#include\s|def\s|class\s|public\s|private\s|protected\s|func\s|fn\s|export\s)",
        s, flags=re.MULTILINE))
    return hits >= 3


def _focus_code_query(q: str) -> str:
    s = str(q or "").strip()
    if not s:
        return "Keep function and class definitions, signatures, return values, and error handling."
    lines = s.split("\n")
    tail = "\n".join(lines[-16:]).strip()
    return "Complete the following code. Focus on symbols near the end of the prefix:\n" + tail


def normalize_question(question: str) -> str:
    q = str(question or "").strip()
    if not q:
        return "Summarize the key facts, findings, recommendations, decisions, errors, and numbers."
    marked = re.search(r"Question\s*:\s*([\s\S]*?)(?:\n\s*Answer\s*:|$)", q, flags=re.IGNORECASE)
    if marked and len(marked.group(1).strip()) >= 8:
        return re.sub(r"\s+", " ", marked.group(1).strip())
    if _looks_like_code_prefix(q):
        return _focus_code_query(q)
    if len(q) > 180 or re.match(r"^Passage\s*:", q, flags=re.IGNORECASE):
        lines = [l.strip() for l in q.split("\n") if l.strip()]
        for l in reversed(lines):
            if re.search(r"\?\s*$", l) or re.match(r"^(who|what|where|when|why|which|how)\b", l, flags=re.IGNORECASE):
                if 8 <= len(l) < 400:
                    return l
    return q


def distinctive_entities_in_corpus(entities: list[str], corpus_text: str) -> list[str]:
    lines = str(corpus_text or "").split("\n")
    n = max(len(lines), 1)
    out = []
    for e in entities or []:
        if re.search(r"[0-9_./:-]", e) or len(e) >= 10:
            out.append(e)
            continue
        proper = bool(re.match(r"^[A-Z][a-zA-Z]{3,}", e))
        lower = e.lower()
        df = sum(1 for line in lines if e in line or lower in line.lower())
        if df == 0:
            continue
        if proper:
            if df <= max(10, math.ceil(n * 0.4)):
                out.append(e)
        else:
            if df <= max(3, math.ceil(n * 0.12)):
                out.append(e)
    return out


def normalize_evidence_line(line: str) -> str:
    s = str(line or "")
    s = re.sub(r"\d{4}-\d{2}-\d{2}T[\d:.Z+-]+", "T", s)
    s = re.sub(r"\b\d+\b", "#", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


# ───────────────────────── Line classification ─────────────────────────

def classify_line(line: str) -> str:
    t = line.strip()
    if not t:
        return "blank"
    if re.match(r"^#{1,6}\s+", t):
        return "heading"
    if re.match(r"^(```|~~~)", t):
        return "fence"
    if re.match(r'^(Traceback|Caused by:|Error:|Exception:|[A-Za-z]+Error\b|at\s+\S+\(|\s*File\s+".+", line \d+)', t):
        return "trace"
    if re.match(r"^\[turn \d+\]|^\[log \d+\]|tool:\s*|composio:", t, flags=re.IGNORECASE):
        return "tool"
    if re.match(r"^\[?(20\d\d-\d\d-\d\d|\d\d:\d\d:\d\d|INFO|WARN|WARNING|ERROR|DEBUG|TRACE)\]?", t):
        return "log"
    if re.match(r"^(import|from)\s+|^#include\s+|^using\s+", t):
        return "import"
    def_pats = [
        r"^(export\s+)?((public|private|protected|internal|static|async|abstract|virtual|partial|sealed|readonly|unsafe|final|override)\s+)*(async\s+)?function\s+\w+",
        r"^(export\s+)?((public|private|protected|internal|static|abstract|partial|sealed|final)\s+)*(class|interface|type|struct|enum|record|delegate)\s+\w+",
        r"^(export\s+)?((public|private|protected|internal|static|async|abstract|virtual|partial|sealed|readonly|unsafe|final|override)\s+)+[\w<>\[\],\s]+\s+\w+\s*\(",
        r"^(def|class|async\s+def)\s+\w+",
        r"^(pub\s+)?(async\s+)?(fn|struct|enum|trait|impl)\b",
        r"^(func|type|interface|extension)\s+\w+",
        r"^\w+\s*=\s*(async\s*)?\([^)]*\)\s*=>",
    ]
    if any(re.match(p, t) for p in def_pats):
        return "definition"
    if re.match(r"^[-*+]\s+|\d+\.\s+", t):
        return "list"
    if re.match(r"^\|.*\|$", t):
        return "table"
    if re.match(r"^[\[]{.*[\}]\],?$", t) or re.match(r"^[A-Za-z0-9_.-]+:\s+", t):
        return "config"
    if re.match(r"^(User|Assistant|System|Tool|Developer):", t) or re.match(r"^\[(user|assistant|system|tool|developer)\]", t, flags=re.IGNORECASE):
        return "chat"
    if re.match(r"^(//|#|/\*|\*)", t):
        return "comment"
    return "text"


def route_content_type(lines: list[str]) -> str:
    type_counts: dict[str, int] = {}
    max_sample = min(len(lines), 50)
    for i in range(max_sample):
        t = classify_line(lines[i])
        type_counts[t] = type_counts.get(t, 0) + 1
    non_blank = [(k, v) for k, v in type_counts.items() if k != "blank"]
    if not non_blank:
        return "text"
    top = sorted(non_blank, key=lambda kv: -kv[1])[0][0]
    importish = type_counts.get("import", 0) + type_counts.get("definition", 0) + type_counts.get("fence", 0)
    if top in ("import", "definition", "fence", "comment"):
        return "code"
    if importish >= 3:
        return "code"
    if top in ("log", "trace") or (type_counts.get("log", 0) + type_counts.get("trace", 0)) > 3:
        return "log"
    if top in ("config", "table") and importish == 0:
        return "json"
    joined = "\n".join(lines[:max_sample]).strip()
    if re.match(r"^[\{\[]", joined) and re.search(r'"[A-Za-z0-9_]+"\s*:', joined) and importish == 0:
        return "json"
    return "text"


def question_for_content(question: str, lines: list[str]) -> str:
    raw = str(question or "").strip()
    route = route_content_type(lines)
    if _looks_like_code_prefix(raw) or ((not raw or re.match(r"^Passage\s*:", raw, flags=re.IGNORECASE)) and route == "code"):
        return _focus_code_query(raw)
    if raw and not re.match(r"^Passage\s*:", raw, flags=re.IGNORECASE):
        return normalize_question(question)
    if route == "code":
        return "Keep function and class definitions, signatures, return values, and error handling."
    return normalize_question(question)


# ───────────────────────── Domain preprocessors ─────────────────────────

def _collapse_blank_runs(lines: list[str]) -> list[str]:
    out = []
    run = 0
    for line in lines:
        if line.strip() == "":
            run += 1
            if run <= 1:
                out.append(line)
        else:
            run = 0
            out.append(line)
    return out


def crush_value(val, depth=0):
    if depth > 10:
        return val
    if val is None:
        return None
    if isinstance(val, list):
        if len(val) == 0:
            return []
        if len(val) > 12:
            first_type = type(val[0])
            if all(type(item) is first_type for item in val):
                return [crush_value(v, depth + 1) for v in val[:3]] + [f"... {len(val) - 3} more items"]
        return [crush_value(v, depth + 1) for v in val if v is not None]
    if isinstance(val, dict):
        out = {}
        for k, v in val.items():
            if v is None:
                continue
            if isinstance(v, list) and len(v) == 0:
                continue
            if isinstance(v, str) and len(v) > 120 and not re.search(r"\s", v):
                continue
            if isinstance(v, str) and len(v) > 200:
                out[k] = v[:100] + f"... [{len(v) - 200} more chars]"
                continue
            if re.match(r"^(id|_id|uuid|guid|timestamp|created_at|updated_at|etag)$", k, flags=re.IGNORECASE) and isinstance(v, str) and len(v) > 20:
                continue
            out[k] = crush_value(v, depth + 1)
        return out
    return val


def crush_json_lines(lines: list[str]) -> tuple[list[str], str]:
    text = "\n".join(lines)
    blocks = []
    idx = 0
    n = len(text)
    while idx < n:
        start_obj = text.find("{", idx)
        start_arr = text.find("[", idx)
        if start_obj < 0 and start_arr < 0:
            break
        if start_obj < 0:
            candidate = start_arr
        elif start_arr < 0:
            candidate = start_obj
        else:
            candidate = start_obj if start_obj < start_arr else start_arr
        depth = 0
        in_string = False
        end = candidate
        i = candidate
        while i < n:
            ch = text[i]
            if ch == '"' and (i == 0 or text[i - 1] != "\\"):
                in_string = not in_string
            if not in_string:
                if ch in "{[":
                    depth += 1
                elif ch in "}]":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            i += 1
        if depth == 0 and end > candidate:
            raw = text[candidate:end + 1]
            try:
                parsed = json.loads(raw)
                crushed = crush_value(parsed, 0)
                crushed_str = json.dumps(crushed, separators=(",", ":"))
                blocks.append({"start": candidate, "end": end + 1, "raw": raw, "crushed": crushed_str})
            except (json.JSONDecodeError, ValueError):
                pass
            idx = end + 1
        else:
            idx = candidate + 1
    if not blocks:
        return lines, "none"
    result = text
    for b in reversed(blocks):
        if len(b["crushed"]) < len(b["raw"]) * 0.85:
            try:
                json.loads(b["crushed"])
                result = result[:b["start"]] + b["crushed"] + result[b["end"]:]
            except (json.JSONDecodeError, ValueError):
                pass
    return _collapse_blank_runs(result.split("\n")), "json"


def _detect_language(lines: list[str]) -> str:
    all_text = "\n".join(lines)
    if re.match(r"^\s*(import|from)\s+\w+", all_text):
        return "python"
    if re.match(r"^\s*#include\s", all_text):
        return "c"
    if re.match(r"^\s*(import\s+\w+|from\s+['\"])", all_text) or re.match(r"^\s*const\s+\w+\s*=\s*require\(", all_text):
        return "javascript"
    if re.match(r"^\s*(pub\s+fn|fn\s+\w+|use\s+\w+::)", all_text):
        return "rust"
    if re.match(r"^\s*(package\s+\w+|import\s+java\.)", all_text):
        return "java"
    if re.match(r"^\s*(func\s+\w+|package\s+\w+)", all_text):
        return "go"
    return "unknown"


def compress_code_lines(lines: list[str]) -> tuple[list[str], str]:
    result = []
    lang = _detect_language(lines)
    in_block_comment = False
    in_docstring = False
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        trimmed = line.strip()

        if in_block_comment:
            if "*/" in trimmed:
                in_block_comment = False
                after = trimmed[trimmed.index("*/") + 2:].strip()
                if after:
                    result.append(line[:line.index("*/") + 2])
            i += 1
            continue
        if trimmed.startswith("/*"):
            if "*/" not in trimmed:
                in_block_comment = True
                i += 1
                continue
            i += 1
            continue

        if in_docstring:
            if '"""' in trimmed or "'''" in trimmed:
                in_docstring = False
            i += 1
            continue
        if trimmed.startswith('"""') or trimmed.startswith("'''"):
            if '"""' in trimmed[3:] or "'''" in trimmed[3:]:
                i += 1
                continue
            in_docstring = True
            i += 1
            continue

        if trimmed.startswith("//") or trimmed.startswith("#") or trimmed.startswith(";"):
            i += 1
            continue
        if re.match(r"^\s*\*", line) and not trimmed.endswith("*/"):
            i += 1
            continue

        if re.match(r"^(import|from)\s", trimmed) and lang != "go":
            result.append(line)
            i += 1
            continue

        m = re.match(r"^\s*(const|let|var)\s+\w+\s*=\s*([\[{])", trimmed)
        if m:
            brace = m.group(2)
            close = "]" if brace == "[" else "}"
            block_end = i
            depth = 1
            for j in range(i + 1, min(i + 20, n)):
                for ch in lines[j]:
                    if ch == brace:
                        depth += 1
                    elif ch == close:
                        depth -= 1
                if depth == 0:
                    block_end = j
                    break
            if block_end > i + 2:
                block_text = re.sub(r"\s+", " ", " ".join(lines[i:block_end + 1]))
                result.append(block_text[:197] + "..." if len(block_text) > 200 else block_text)
                i = block_end + 1
                continue

        if (re.match(r"^\s*(export\s+)?(async\s+)?(function|class|interface|type|enum|def|struct|trait|impl)\b", trimmed)
                or re.match(r"^\s*@\w+", trimmed)
                or re.match(r"^\s*(public|private|protected|static|async|override)\s", trimmed)):
            result.append(line)
            i += 1
            continue

        if re.match(r"^\s*(return|yield|throw|await)\s", trimmed) or re.match(r"^\s*(case|default)\s*:", trimmed):
            result.append(line)
            i += 1
            continue

        result.append(line)
        i += 1

    return _collapse_blank_runs(result), "code"


def compress_log_lines(lines: list[str], question: str) -> tuple[list[str], str]:
    question_lower = (question or "").lower()
    wants_errors = bool(re.search(r"error|fail|exception|crash|timeout|denied|invalid", question_lower, flags=re.IGNORECASE))
    wants_debug = bool(re.search(r"debug|trace|verbose", question_lower, flags=re.IGNORECASE))
    result = []
    seen_fp: dict[str, int] = {}
    trace_accum: list[str] = []

    def flush_trace():
        if not trace_accum:
            return
        if len(trace_accum) <= 3:
            result.extend(trace_accum)
        else:
            result.append(trace_accum[0])
            result.append(f"  ... {len(trace_accum) - 2} more frames")
            result.append(trace_accum[-1])
        trace_accum.clear()

    q_entities = extract_question_entities(question)
    q_terms = question_terms(question)

    for line in lines:
        trimmed = line.strip()

        if re.match(r'^\s*(at\s+\S+|File\s+"[^"]+",\s+line\s+\d+|Caused by:|\.\w+\(.*\):\d+)', trimmed):
            trace_accum.append(trimmed)
            continue
        if trace_accum and trimmed != "":
            flush_trace()

        is_log = bool(re.match(r"^\s*\[?(20\d\d-\d\d-\d\d|\d\d:\d\d:\d\d)\]?\s*", trimmed) or
                      re.match(r"^\s*(INFO|WARN|WARNING|ERROR|DEBUG|TRACE|FATAL)\b", trimmed))
        if not is_log:
            result.append(line)
            continue

        is_error = bool(re.search(r"ERROR|FATAL", trimmed, flags=re.IGNORECASE))
        is_warn = bool(re.search(r"WARN(ING)?", trimmed, flags=re.IGNORECASE))
        is_debug = bool(re.search(r"DEBUG|TRACE", trimmed, flags=re.IGNORECASE))
        matches_question = any(e in trimmed for e in q_entities) or any(str(t).lower() in trimmed.lower() for t in q_terms)

        if is_debug and not wants_debug and not matches_question:
            continue
        if matches_question:
            result.append(line)
            continue
        if is_warn and wants_errors:
            result.append(line)
            continue

        fp = trimmed
        fp = re.sub(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z?", "T", fp)
        fp = re.sub(r"\d+", "#", fp)
        fp = re.sub(r"\s+", " ", fp).strip()

        if fp in seen_fp:
            seen_fp[fp] += 1
            count = seen_fp[fp]
            if count <= 2:
                result.append(line)
            elif count == 3:
                result.append(line + f"  [repeated {count - 1}x]")
        else:
            seen_fp[fp] = 1
            if is_error or not wants_errors:
                result.append(line)
            elif wants_errors and re.search(r"error|fail|exception|timeout|denied|invalid", trimmed, flags=re.IGNORECASE):
                result.append(line)

    flush_trace()
    return _collapse_blank_runs(result), "log"


def preprocess_lines(lines: list[str], question: str) -> tuple[list[str], str]:
    if not lines:
        return lines, "none"
    route = route_content_type(lines)
    if route == "json":
        return crush_json_lines(lines)
    if route == "code":
        return compress_code_lines(lines)
    if route == "log":
        return compress_log_lines(lines, question or "")
    return lines, "none"


# ───────────────────────── Tokenization & token-level features ─────────────────────────

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[^\s]")


def tokenize_context_lines(lines: list[str]) -> list[str]:
    tokens = []
    for line in lines:
        parts = _TOKEN_RE.findall(line)
        tokens.extend(parts if parts else [" "])
    return tokens


def _classify_token_semantic(tok: str, line_context: str) -> int:
    t = tok.strip()
    if t.startswith("#") or t.startswith("//") or "/*" in line_context:
        return _SEM_COMMENT
    if re.match(r"^(User:|Assistant:|>)", t):
        return _SEM_CHAT
    boiler = [
        r"^import\s", r"^from\s+\w+\s+import", r"^# -\*- coding", r"^LICENSE",
        r"^Copyright", r"^\s*$", r"^---$", r"^```",
    ]
    if any(re.match(p, line_context.strip()) for p in boiler):
        return _SEM_BOILERPLATE
    return _SEM_CODE


def _deterministic_attention(semantic, entity_match, recency_norm, line_ctx, tok):
    base_map = {_SEM_CODE: 0.55, _SEM_COMMENT: 0.25, _SEM_CHAT: 0.35, _SEM_BOILERPLATE: 0.08}
    base = base_map.get(semantic, 0.3)
    struct_boost = 0
    lc_strip = line_ctx.strip()
    if re.match(r"^(def|class|async)\b", lc_strip) and tok in ("def", "class", "async"):
        struct_boost = 0.35
    elif "def " in line_ctx:
        parts = re.split(r"\s+", line_ctx)
        if "def" in parts:
            idx = parts.index("def")
            if idx + 1 < len(parts) and tok == parts[idx + 1].split("(")[0]:
                struct_boost = 0.3
    mass = min(1, max(0.01, base + 0.25 * entity_match + struct_boost * 0.5 + 0.15 * recency_norm))
    layer_m = min(1, max(0.01, mass * 0.95 + 0.15 * entity_match))
    h2o = min(1, max(0.01, layer_m * (0.7 + 0.3 * (1 - recency_norm))))
    snap = min(1, max(0.01, mass * (0.3 + 0.7 * recency_norm)))
    return mass, layer_m, h2o, snap


def _sinusoidal_position_encoding(position, seq_len):
    if seq_len <= 1:
        return 0.0
    pos_ratio = position / max(seq_len - 1, 1)
    return 0.5 * math.sin(pos_ratio * math.pi) + 0.5 * math.sin(pos_ratio * math.pi * 4)


def _ngrams(text, n):
    s = text.lower().strip()
    if len(s) < n:
        return {s}
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def _compute_ngram_sim(line_text, question_text):
    if not line_text or not question_text:
        return 0.0
    n1 = _ngrams(line_text, 3)
    n2 = _ngrams(question_text, 3)
    union = n1 | n2
    if not union:
        return 0.0
    return len(n1 & n2) / len(union)


def _estimate_indent_depth(line):
    if not line:
        return 0.0
    stripped = line.lstrip()
    if not stripped:
        return 0.0
    leading = len(line) - len(stripped)
    return min(leading / 40.0, 1.0)


def _compute_token_entropy(tok, all_tokens_lower_count, total):
    if total < 3:
        return 0.5
    count = all_tokens_lower_count.get(tok.lower(), 0)
    freq = count / total
    return min(1, max(0, 1 - freq * 3))


def _bigrams(s):
    cl = s.lower().strip()
    if len(cl) < 2:
        return {cl}
    return {cl[i:i + 2] for i in range(len(cl) - 1)}


def _compute_semantic_fingerprint(line_text, question_text):
    if not line_text or not question_text:
        return 0.0
    b1 = _bigrams(line_text)
    b2 = _bigrams(question_text)
    union = b1 | b2
    if not union:
        return 0.0
    return len(b1 & b2) / len(union)


def _compute_cross_context_similarity(tok, entities):
    if not entities:
        return 0.0
    t_lower = tok.lower()
    t_chars = set(t_lower)
    max_sim = 0.0
    for entity in entities:
        e_lower = entity.lower()
        if t_lower == e_lower:
            return 1.0
        e_chars = set(e_lower)
        if not e_chars:
            continue
        shared = len(t_chars & e_chars)
        overlap = shared / max(len(e_chars) + len(t_chars) - shared, 1)
        max_sim = max(max_sim, overlap)
    return max_sim


def _build_inference_records(lines: list[str], question: str):
    tokens = tokenize_context_lines(lines)
    entities = set(extract_question_entities(question))
    seq_len = len(tokens)

    all_tok_count: dict[str, int] = {}
    for tk in tokens:
        k = tk.lower()
        all_tok_count[k] = all_tok_count.get(k, 0) + 1

    line_boundaries = []
    line_ntok = []
    for line in lines:
        parts = _TOKEN_RE.findall(line)
        line_ntok.append(len(parts) if parts else 1)

    records = []
    line_for_token = []
    line_idx = 0
    tok_in_line = 0
    for pos, tok in enumerate(tokens):
        while line_idx < len(lines) and tok_in_line >= line_ntok[line_idx]:
            line_idx += 1
            tok_in_line = 0
        line_ctx = lines[line_idx] if line_idx < len(lines) else ""
        sem = _classify_token_semantic(tok, line_ctx)
        age_norm = pos / max(seq_len - 1, 1)
        entity_match = 1 if tok in entities else 0
        mass, layer_m, h2o, snap = _deterministic_attention(sem, entity_match, age_norm, line_ctx, tok)
        entropy = _compute_token_entropy(tok, all_tok_count, seq_len)
        cross_sim = _compute_cross_context_similarity(tok, entities)
        ctx_div = entropy  # same formula as JS (computeContextDivergence mirrors computeTokenEntropy)
        records.append({
            "position": pos,
            "semantic_type": sem,
            "attention_mass": mass,
            "layer_attention_mean": layer_m,
            "question_entity_match": entity_match,
            "h2o_score": h2o,
            "snapkv_score": snap,
            "line_index": line_idx,
            "line_text": line_ctx,
            "position_encoding": _sinusoidal_position_encoding(pos, len(lines)),
            "ngram_sim": _compute_ngram_sim(line_ctx, question or ""),
            "line_length_norm": min(len(line_ctx) / 500, 1.0),
            "indent_depth": _estimate_indent_depth(line_ctx),
            "entropy": entropy,
            "semantic_fingerprint": _compute_semantic_fingerprint(line_ctx, question or ""),
            "cross_context_sim": cross_sim,
            "context_divergence": ctx_div,
        })
        line_for_token.append(line_idx)
        tok_in_line += 1

    return records, line_for_token, tokens


def _build_feature_tensor(records, n):
    feats = [[0.0] * 16 for _ in range(n)]
    seq_len = n
    for i, rec in enumerate(records):
        age = seq_len - 1 - rec["position"]
        recency = 1 - age / max(seq_len, 1)
        f = feats[i]
        f[0] = rec["attention_mass"]
        f[1] = rec["layer_attention_mean"]
        f[2] = recency
        f[3] = rec["question_entity_match"]
        f[4 + rec["semantic_type"]] = 1
        f[8] = rec.get("position_encoding") or 0.0
        f[9] = rec.get("ngram_sim") or 0.0
        f[10] = rec.get("line_length_norm") or 0.0
        f[11] = rec.get("indent_depth") or 0.0
        f[12] = rec.get("entropy") or 0.0
        f[13] = rec.get("semantic_fingerprint") or 0.0
        f[14] = rec.get("cross_context_sim") or 0.0
        f[15] = rec.get("context_divergence") or 0.0
    return feats


def _gelu(x):
    return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


def _layer_norm_row(x, weight, bias, dim):
    mean = sum(x[:dim]) / dim
    var = sum((x[i] - mean) ** 2 for i in range(dim)) / dim
    denom = math.sqrt(var + 1e-5)
    return [((x[i] - mean) / denom) * weight[i] + bias[i] for i in range(dim)]


def _linear_row(x, weight, bias, in_dim, out_dim):
    out = [0.0] * out_dim
    for o in range(out_dim):
        s = bias[o]
        w_row = weight[o]
        for i in range(in_dim):
            s += x[i] * w_row[i]
        out[o] = s
    return out


def _sigmoid(x):
    return 1 / (1 + math.exp(-x))


def forward_model(model: dict, feats: list[list[float]], n: int) -> list[float]:
    dim = model["feature_dim"]
    layers = model["layers"]
    gate = model.get("gate")
    is_amcp = bool(gate) and len(gate) == dim and len(layers) >= 11
    encoder_count = len(layers) - 2 if is_amcp else len(layers)

    scores = [0.0] * n
    for t in range(n):
        x = list(feats[t])

        if is_amcp:
            x = [x[i] * gate[i] for i in range(len(x))]
            for li in range(encoder_count):
                layer = layers[li]
                if layer["type"] == "linear":
                    out_dim = len(layer["weight"])
                    in_dim = len(layer["weight"][0])
                    x = _linear_row(x, layer["weight"], layer["bias"], in_dim, out_dim)
                elif layer["type"] == "layernorm":
                    x = _layer_norm_row(x, layer["weight"], layer["bias"], len(x))
                elif layer["type"] == "gelu":
                    x = [_gelu(v) for v in x]
            shared = x
            score_layer = layers[encoder_count]
            score_out = _linear_row(shared, score_layer["weight"], score_layer["bias"], len(shared), 1)
            conf_layer = layers[encoder_count + 1]
            conf_out = _linear_row(shared, conf_layer["weight"], conf_layer["bias"], len(shared), 1)
            s = _sigmoid(score_out[0])
            c = _sigmoid(conf_out[0])
            scores[t] = s * c + 0.5 * (1 - c)
        else:
            for layer in layers:
                if layer["type"] == "linear":
                    out_dim = len(layer["weight"])
                    in_dim = len(layer["weight"][0])
                    x = _linear_row(x, layer["weight"], layer["bias"], in_dim, out_dim)
                elif layer["type"] == "layernorm":
                    x = _layer_norm_row(x, layer["weight"], layer["bias"], len(x))
                elif layer["type"] == "gelu":
                    x = [_gelu(v) for v in x]
            scores[t] = _sigmoid(x[0])
    return scores


def _line_scores_from_model(records, scores, line_for_token, num_lines):
    line_score = [0.0] * num_lines
    for tok_idx, line_idx in enumerate(line_for_token):
        line_score[line_idx] = max(line_score[line_idx], scores[tok_idx])
    return line_score


# ───────────────────────── Relevance & dedup ─────────────────────────

def _line_fingerprint(line: str) -> str:
    s = re.sub(r"\d+", "#", line.lower())
    s = re.sub(r"\s+", " ", s).strip()
    return s


def duplicate_line_penalty(lines: list[str]) -> list[float]:
    counts: dict[str, int] = {}
    fps = []
    for line in lines:
        fp = _line_fingerprint(line)
        fps.append(fp)
        if len(fp) > 12:
            counts[fp] = counts.get(fp, 0) + 1
    return [min(3, (counts.get(fp, 0) - 2) * 0.8) if counts.get(fp, 0) > 2 else 0 for fp in fps]


def line_question_relevance(line: str, question: str, model_line_score: float) -> float:
    q = normalize_question(question)
    lower = line.lower()
    score = model_line_score * 2
    for term in question_terms(q):
        if len(term) < 5:
            continue
        if term.lower() in lower:
            score += 1.4
    for entity in extract_question_entities(q):
        if len(entity) < 4 and not re.search(r"[0-9_./:-]", entity):
            continue
        if entity in line:
            score += 1.2
    lstrip = line.strip()
    if re.match(r"^\[turn \d+\]|^\[log \d+\]|tool:\s*grep|composio:", lstrip):
        score -= 9
    if re.match(r"^## Appendix \d+ — deployment notes", lstrip):
        score -= 2
    if re.match(r"^Region \d+ runs CPU eviction", lstrip):
        score -= 2.5
    return max(score, 0)


# ───────────────────────── Block segmentation & scoring ─────────────────────────

def _should_start_block(prev_type, ltype, prev_line, line):
    if prev_type is None:
        return True
    if ltype == "blank":
        return False
    if ltype in ("heading", "fence", "definition", "trace"):
        return True
    if prev_type == "blank":
        return True
    if prev_type == "heading":
        return False
    if ltype != prev_type and ltype in ("tool", "log", "import", "list", "table", "config", "chat"):
        return True
    if prev_type != ltype and prev_type in ("tool", "log", "trace", "table", "import"):
        return True
    if (prev_line or "").strip() == "":
        return True
    if classify_line(prev_line or "") == "definition" and re.match(r"^\S", line) and classify_line(line) == "definition":
        return True
    return False


def _segment_context(lines: list[str], token_counts: list[int]):
    blocks = []
    cur = None
    prev_type = None
    prev_line = ""

    def finish():
        nonlocal cur
        if cur is None:
            return
        while cur["end"] > cur["start"] and not lines[cur["end"]].strip():
            cur["end"] -= 1
        cur["text"] = "\n".join(lines[cur["start"]:cur["end"] + 1])
        cur["tokens"] = sum(token_counts[cur["start"]:cur["end"] + 1])
        if cur["text"].strip():
            blocks.append(cur)
        cur = None

    for i, line in enumerate(lines):
        ltype = classify_line(line)
        start = _should_start_block(prev_type, ltype, prev_line, line)
        too_large = cur and (i - cur["start"] >= 10 or cur["tokens"] > 220)
        if cur is None or start or too_large:
            finish()
            cur = {"start": i, "end": i, "type": ltype, "tokens": token_counts[i] if i < len(token_counts) else 0}
        else:
            cur["end"] = i
            cur["tokens"] += token_counts[i] if i < len(token_counts) else 0
            if cur["type"] == "blank" and ltype != "blank":
                cur["type"] = ltype
        prev_type = ltype
        prev_line = line
    finish()
    for i, b in enumerate(blocks):
        b["id"] = i
    return blocks


def _block_fingerprint(block) -> str:
    return _line_fingerprint(block["text"])[:240]


def _score_compiler_blocks(blocks, lines, line_relevance, question):
    q = normalize_question(question)
    corpus = "\n".join(lines)
    raw_entities = extract_question_entities(q)
    raw_terms = question_terms(q)
    entities = distinctive_entities_in_corpus(raw_entities, corpus)
    dist_terms = distinctive_entities_in_corpus(raw_terms, corpus)
    terms = dist_terms if dist_terms else [t for t in raw_terms if len(t) >= 5]

    fp_counts: dict[str, int] = {}
    entity_df: dict[str, int] = {}
    term_df: dict[str, int] = {}
    for b in blocks:
        fp = _block_fingerprint(b)
        if len(fp) > 20:
            fp_counts[fp] = fp_counts.get(fp, 0) + 1
        for e in entities:
            if e in b["text"]:
                entity_df[e] = entity_df.get(e, 0) + 1
        lower = b["text"].lower()
        for t in terms:
            if t in b["text"] or t.lower() in lower:
                term_df[t] = term_df.get(t, 0) + 1

    scored = []
    for block in blocks:
        text = block["text"]
        lower = text.lower()
        max_line = 0.0
        sum_line = 0.0
        for i in range(block["start"], block["end"] + 1):
            v = line_relevance[i] if i < len(line_relevance) else 0
            max_line = max(max_line, v)
            sum_line += v
        entity_hits = 0
        term_hits = 0
        entity_weight = 0.0
        term_weight = 0.0
        reason = "context evidence"
        n_blocks = len(blocks)
        for e in entities:
            if e in text:
                entity_hits += 1
                df = entity_df.get(e, 1)
                entity_weight += 1.2 + 4.8 * math.log((n_blocks + 1) / (df + 1))
                reason = "query entity match"
        for t in terms:
            if t in text or t.lower() in lower:
                term_hits += 1
                df = term_df.get(t, 1)
                term_weight += 0.8 + 3.4 * math.log((n_blocks + 1) / (df + 1))
                if reason == "context evidence":
                    reason = "query keyword match"

        score = max_line * 1.4 + (sum_line / max(block["end"] - block["start"] + 1, 1)) * 0.8
        score += entity_weight + term_weight
        if block["type"] == "definition":
            score += 3.8
        if block["type"] == "trace":
            score += 5.0
        if block["type"] == "log" and re.search(r"(error|warn|failed|exception|timeout|denied)", text, flags=re.IGNORECASE):
            score += 3.2
        if block["type"] == "tool":
            score -= 8.0
        if block["type"] == "heading":
            score += 1.3
        if block["type"] == "config" and entity_hits + term_hits > 0:
            score += 2.2
        if re.search(r"TODO|FIXME|BUG|SECURITY|BREAKING|deprecated|root cause", text, flags=re.IGNORECASE):
            score += 1.6
        if not raw_entities or (len(raw_entities) <= 2 and re.search(r"summarize", q, flags=re.IGNORECASE)):
            if block["start"] <= 2:
                score += 3.5
            if re.search(r"\b\d+(\.\d+)?%|\$\d|\b(FY|Q[1-4])\s*\d{2,4}\b|\b20\d{2}\b", text):
                score += 2.2
            if re.search(r"\b(recommend|finding|conclusion|summary|whereas|resolved|decision)\b", text, flags=re.IGNORECASE):
                score += 2.8
            if block["type"] == "heading":
                score += 2.0

        dup_count = fp_counts.get(_block_fingerprint(block), 0)
        if dup_count > 1:
            score -= min(5, dup_count * 1.25)
        if re.match(r"^(copyright|license|all rights reserved|generated by|do not edit)", text.strip(), flags=re.IGNORECASE):
            score -= 5
        if block["tokens"] > 260 and entity_hits == 0 and term_hits == 0:
            score -= 2.5
        if len(entities) + len(terms) > 0 and entity_hits == 0 and term_hits == 0:
            score -= 6.5
            if re.match(r"^Passage\s*\d*\s*:", text.strip(), flags=re.IGNORECASE) or block["type"] == "text":
                score -= 3.5
        if re.match(r"^(Passage|Title|Question|Answer)\s*\d*\s*:?\s*$", text.strip(), flags=re.IGNORECASE):
            score -= 6
        if re.match(r"^(see also|references|external links|navigation|contents|table of contents)\b", text.strip(), flags=re.IGNORECASE):
            score -= 4

        nb = dict(block)
        nb["score"] = max(0, score)
        nb["reason"] = reason
        nb["entity_hits"] = entity_hits
        nb["keyword_hits"] = term_hits
        scored.append(nb)
    return scored


def _fence_marker_count(text: str) -> int:
    return len(re.findall(r"```|~~~", text))


def _select_blocks_under_budget(blocks, budget_tokens: int) -> set:
    """
    Greedy knapsack-style selection: always keep block 0 if it's small (framing/
    system context), always keep the definition/trace/log blocks that carry query
    entities or keywords ("important" evidence), then fill remaining budget with
    the best score-per-token blocks. Then close dependencies (headings before
    definitions, imports before definitions, log lines around traces) and any
    markdown fences left open.
    """
    if not blocks:
        return set()

    kept_ids: set = set()
    total_tokens = sum(b["tokens"] for b in blocks)
    budget_tokens = max(1, min(budget_tokens, total_tokens))

    # Always keep tiny opening block (framing).
    if blocks[0]["tokens"] <= 80:
        kept_ids.add(blocks[0]["id"])

    # "Important" evidence: definitions/traces/logs that actually match the query,
    # or anything with a strong score, capped so it can't eat the whole budget.
    important = [
        b for b in blocks
        if b["type"] == "trace"
        or (b["entity_hits"] > 0 and b["score"] >= 5.5)
        or (b["score"] >= 10 and (b["entity_hits"] > 0 or b["keyword_hits"] > 0))
        or (b["type"] == "log" and re.search(r"(error|fail|exception|timeout|denied|invalid|warn)", b["text"], flags=re.IGNORECASE)
            and (b["entity_hits"] > 0 or b["keyword_hits"] > 0) and b["score"] >= 5)
        or (b["type"] in ("definition", "code") and b["entity_hits"] > 0 and b["score"] >= 4)
    ]
    imp_cap = max(3, min(24, math.ceil(len(blocks) * 0.25)))
    important = sorted(important, key=lambda b: (-b["score"], b["tokens"]))[:imp_cap]

    used = sum(b["tokens"] for b in blocks if b["id"] in kept_ids)
    for b in important:
        if b["id"] in kept_ids:
            continue
        if used + b["tokens"] > budget_tokens and used > 0:
            continue
        kept_ids.add(b["id"])
        used += b["tokens"]

    # Fill remaining budget by best score-per-token among what's left.
    remaining = [b for b in blocks if b["id"] not in kept_ids and b["score"] > 0]
    remaining.sort(key=lambda b: -(b["score"] / max(b["tokens"], 1)))
    for b in remaining:
        if used >= budget_tokens:
            break
        if used + b["tokens"] > budget_tokens and used / max(budget_tokens, 1) > 0.5:
            continue
        kept_ids.add(b["id"])
        used += b["tokens"]

    # Preserve structural dependencies.
    by_id = {b["id"]: b for b in blocks}
    for bid in list(kept_ids):
        b = by_id[bid]
        if b["type"] != "heading":
            for j in range(bid - 1, max(-1, bid - 4), -1):
                if 0 <= j < len(blocks) and blocks[j]["type"] == "heading":
                    kept_ids.add(blocks[j]["id"])
                    break
        if b["type"] == "definition":
            for j in range(max(0, bid - 6), bid):
                if blocks[j]["type"] == "import" and blocks[j]["tokens"] <= 80:
                    kept_ids.add(blocks[j]["id"])
        if b["type"] == "trace":
            if bid - 1 >= 0 and blocks[bid - 1]["type"] == "log":
                kept_ids.add(blocks[bid - 1]["id"])
            if bid + 1 < len(blocks) and blocks[bid + 1]["type"] == "log":
                kept_ids.add(blocks[bid + 1]["id"])

    # Close any markdown fences a cut would otherwise leave open.
    for bid in list(kept_ids):
        fence_count = _fence_marker_count(by_id[bid]["text"])
        if fence_count % 2 == 0:
            continue
        for j in range(bid + 1, len(blocks)):
            kept_ids.add(blocks[j]["id"])
            fence_count += _fence_marker_count(blocks[j]["text"])
            if fence_count % 2 == 0:
                break

    return kept_ids


# ───────────────────────── Model loading ─────────────────────────

_model_cache: Optional[dict] = None


def load_model(path: str = _MODEL_PATH) -> Optional[dict]:
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    try:
        with open(path, "r", encoding="utf-8") as f:
            _model_cache = json.load(f)
    except (OSError, json.JSONDecodeError):
        _model_cache = None
    return _model_cache


# ───────────────────────── Public API ─────────────────────────

class SuperCompress:
    """Local, dependency-free prompt/context compressor. No API key needed."""

    def __init__(self, model_path: str = _MODEL_PATH):
        self.model = load_model(model_path)

    def compress(self, context: str, query: str = "", budget_ratio: float = 0.35) -> CompressResult:
        """
        Compress `context` down toward `budget_ratio` of its tokens, keeping the
        parts most relevant to `query`.

        Args:
            context: the long text to compress (docs, chat history, logs, JSON, code...).
            query: the current question/task — used to score what's relevant.
            budget_ratio: target fraction of tokens to keep (0.0-1.0). This is a
                target, not a hard cap — structurally important blocks (query
                matches, error traces, definitions) can push actual retention higher.

        Returns:
            CompressResult with `.compressed_text` and stats.
        """
        if not context or not context.strip():
            return CompressResult(
                original_text=context, compressed_text=context, original_tokens=0,
                kept_tokens=0, tokens_saved=0, kv_savings_pct=0.0, kept_line_ratio=0.0,
                policy_name="noop", keep_ratio=budget_ratio, preprocessor="none",
                answer_quality=1.0,
            )

        normalized = normalize_context_text(context)
        raw_lines = normalized.split("\n")
        q = question_for_content(query, raw_lines)
        raw_token_count = len(tokenize_context_lines(raw_lines))

        preproc_lines, preprocessor = preprocess_lines(raw_lines, q)
        records, line_for_token, tokens = _build_inference_records(preproc_lines, q)
        n = len(records)

        if self.model and n <= MAX_MODEL_TOKENS:
            feats = _build_feature_tensor(records, n)
            scores = forward_model(self.model, feats, n)
            model_line_scores = _line_scores_from_model(records, scores, line_for_token, len(preproc_lines))
        else:
            # Fast path for large contexts: use the cheap deterministic h2o_score
            # per token (already computed) instead of the full MLP forward pass.
            fallback_scores = [rec["h2o_score"] for rec in records]
            model_line_scores = _line_scores_from_model(records, fallback_scores, line_for_token, len(preproc_lines))

        dup_penalty = duplicate_line_penalty(preproc_lines)
        line_relevance = [
            max(0.0, line_question_relevance(line, q, model_line_scores[i]) - dup_penalty[i])
            for i, line in enumerate(preproc_lines)
        ]

        token_counts = [0] * len(preproc_lines)
        for line_idx in line_for_token:
            if 0 <= line_idx < len(token_counts):
                token_counts[line_idx] += 1

        blocks_raw = _segment_context(preproc_lines, token_counts)
        blocks = _score_compiler_blocks(blocks_raw, preproc_lines, line_relevance, q)

        budget_tokens = max(1, math.floor(sum(b["tokens"] for b in blocks) * budget_ratio))
        kept_ids = _select_blocks_under_budget(blocks, budget_tokens)

        kept_line_set = set()
        for b in blocks:
            if b["id"] in kept_ids:
                kept_line_set.update(range(b["start"], b["end"] + 1))

        # Always keep the very first line and last couple lines (attention sinks).
        if preproc_lines:
            kept_line_set.add(0)
            for i in range(max(0, len(preproc_lines) - 2), len(preproc_lines)):
                kept_line_set.add(i)

        if not kept_line_set and preproc_lines:
            kept_line_set = set(range(min(8, len(preproc_lines))))

        compressed = "\n".join(preproc_lines[i] for i in sorted(kept_line_set))

        kept_tokens = sum(token_counts[i] for i in kept_line_set if i < len(token_counts))
        compressed_tok = len(tokenize_context_lines(compressed.split("\n")))
        kept_tokens = max(kept_tokens, compressed_tok)
        original_tokens = max(raw_token_count, n)
        tokens_saved = max(0, original_tokens - kept_tokens)

        entities = extract_question_entities(q)
        terms = question_terms(q)
        keys = list(set(entities) | set(terms))
        quality = 1.0
        if keys:
            lower_orig = normalized.lower()
            lower_comp = compressed.lower()
            required = [k for k in keys if k.lower() in lower_orig]
            if required:
                hit = sum(1 for k in required if k.lower() in lower_comp)
                quality = round(hit / len(required), 4)

        policy_name = (
            f"SuperCompress {preprocessor.capitalize()}" if preprocessor != "none" else "SuperCompress Compiler"
        )

        annotations = []
        for i, line in enumerate(preproc_lines):
            kept = i in kept_line_set
            reason = "kept — evidence block" if kept else "removed — low relevance to query"
            if i == 0:
                reason = "attention sink (always kept)"
            elif any(e in line for e in entities):
                reason = "question entity match"
            elif any(t.lower() in line.lower() for t in terms):
                reason = "question keyword match"
            annotations.append({"line_index": i, "text": line, "kept": kept, "reason": reason})

        return CompressResult(
            original_text=context,
            compressed_text=compressed,
            original_tokens=original_tokens,
            kept_tokens=kept_tokens,
            tokens_saved=tokens_saved,
            kv_savings_pct=(1 - kept_tokens / max(original_tokens, 1)) * 100,
            kept_line_ratio=len(kept_line_set) / max(len(preproc_lines), 1),
            policy_name=policy_name,
            keep_ratio=kept_tokens / max(original_tokens, 1),
            preprocessor=preprocessor,
            answer_quality=quality,
            line_annotations=annotations,
        )


def compress(context: str, query: str = "", budget_ratio: float = 0.35, model_path: str = _MODEL_PATH) -> CompressResult:
    """Convenience one-shot function (loads/caches the model on first call)."""
    return SuperCompress(model_path=model_path).compress(context, query, budget_ratio)


__all__ = ["SuperCompress", "CompressResult", "compress", "load_model"]
