"""
Optional LangGraph glue for supercompress_local. LangGraph/LangChain message
types are NOT a dependency of this package — imports are lazy and there's a
plain-dict fallback, so this works whether your state stores LangChain
BaseMessage objects or plain {"role": ..., "content": ...} dicts.
"""

from __future__ import annotations

from typing import Any, Optional

from .compressor import SuperCompress


def _role_of(msg: Any) -> str:
    if isinstance(msg, dict):
        return msg.get("role", "user")
    # LangChain BaseMessage subclasses expose `.type` ("human", "ai", "system", "tool")
    role = getattr(msg, "type", None) or getattr(msg, "role", None) or "user"
    return {"human": "user", "ai": "assistant"}.get(role, role)


def _content_of(msg: Any) -> str:
    content = (
        msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
    )
    if isinstance(content, str):
        return content
    return str(content)


def assemble_messages(messages: list) -> tuple[str, str, Optional[Any]]:
    """
    Mirrors `assembleMessages` from your proxy/src/compressor.js:
    the last user/human message becomes the `query`, everything before it
    becomes the `context` to compress, and any leading system message is
    kept aside untouched.

    Returns (context, query, system_message_or_None).
    """
    if not messages:
        return "", "", None

    system_msg = None
    non_system = []
    for m in messages:
        if _role_of(m) == "system":
            system_msg = m
        else:
            non_system.append(m)

    if not non_system:
        return "", "", system_msg

    last = non_system[-1]
    if _role_of(last) == "user":
        query = _content_of(last)
        parts = [f"[{_role_of(m)}]: {_content_of(m)}" for m in non_system[:-1]]
    else:
        parts = [f"[{_role_of(m)}]: {_content_of(m)}" for m in non_system]
        query = "Continue the conversation."

    return "\n\n".join(parts), query, system_msg


def _as_plain_message(m: Any) -> dict:
    if isinstance(m, dict):
        return m

    role = _role_of(m)
    out = {"role": role, "content": _content_of(m)}

    # Preserve tool_call_id for tool-result messages
    tool_call_id = getattr(m, "tool_call_id", None)
    if tool_call_id is not None:
        out["tool_call_id"] = tool_call_id

    # Preserve tool_calls for assistant messages that requested tool use
    tool_calls = getattr(m, "tool_calls", None)
    if tool_calls:
        out["tool_calls"] = tool_calls

    # Preserve name if present (some providers need it, e.g. tool/function name)
    name = getattr(m, "name", None)
    if name is not None:
        out["name"] = name

    return out


def compress_messages(
    messages: list,
    budget_ratio: float = 0.35,
    sc: Optional[SuperCompress] = None,
    min_words: int = 100,
    keep_last_turns: int = 4,
) -> dict:
    """
    Compress a LangGraph-style message list. Returns a dict:
        {"messages": [...], "original_tokens": int, "kept_tokens": int,
         "tokens_saved": int, "savings_pct": float, "skipped": bool}

    keep_last_turns: how many of the most-recent non-system messages are
        passed through completely untouched — never scored, never dropped,
        never summarized. This is a hard guarantee, not a scoring hint: it's
        what stops "the AI loses recent context" regardless of how the
        relevance model scores any individual turn. Only messages older than
        this window are candidates for compression, and that older portion
        still respects budget_ratio exactly as before.

    The returned `messages` list has the same shape as the input (list of
    dicts) with older history collapsed into one system-role summary message,
    the original system prompt (if any) kept first, followed by the last
    `keep_last_turns` messages verbatim (including a trailing assistant
    message, if any — previously only a trailing *user* message survived).
    """
    if not messages:
        return {
            "messages": messages,
            "original_tokens": 0,
            "kept_tokens": 0,
            "tokens_saved": 0,
            "savings_pct": 0.0,
            "skipped": True,
        }

    system_msg = next((m for m in messages if _role_of(m) == "system"), None)
    non_system = [m for m in messages if _role_of(m) != "system"]

    if not non_system:
        return {
            "messages": messages,
            "original_tokens": 0,
            "kept_tokens": 0,
            "tokens_saved": 0,
            "savings_pct": 0.0,
            "skipped": True,
        }

    keep_last_turns = max(1, keep_last_turns)
    recent = non_system[-keep_last_turns:]
    older = non_system[:-keep_last_turns]

    if not older:
        # Nothing old enough to be a compression candidate — return as-is.
        return {
            "messages": messages,
            "original_tokens": 0,
            "kept_tokens": 0,
            "tokens_saved": 0,
            "savings_pct": 0.0,
            "skipped": True,
        }

    context = "\n\n".join(f"[{_role_of(m)}]: {_content_of(m)}" for m in older)
    word_count = len(context.split())
    if word_count < min_words:
        return {
            "messages": messages,
            "original_tokens": word_count,
            "kept_tokens": word_count,
            "tokens_saved": 0,
            "savings_pct": 0.0,
            "skipped": True,
        }

    # The current query, for relevance scoring of the *older* material —
    # the most recent user message anywhere in the conversation, even though
    # it also happens to be one of the untouched `recent` messages now.
    last_user = next((m for m in reversed(non_system) if _role_of(m) == "user"), None)
    query = (
        _content_of(last_user)
        if last_user is not None
        else "Continue the conversation."
    )

    engine = sc or SuperCompress()
    result = engine.compress(context, query=query, budget_ratio=budget_ratio)

    out = []
    if system_msg is not None:
        out.append(_as_plain_message(system_msg))
    out.append(
        {
            "role": "system",
            "content": f"[Compressed earlier context — {result.tokens_saved} tokens saved (~{round(result.kv_savings_pct)}%)]\n\n{result.compressed_text}",
        }
    )
    out.extend(_as_plain_message(m) for m in recent)

    return {
        "messages": out,
        "original_tokens": result.original_tokens,
        "kept_tokens": result.kept_tokens,
        "tokens_saved": result.tokens_saved,
        "savings_pct": round(result.kv_savings_pct, 1),
        "skipped": False,
    }


def make_compression_node(
    budget_ratio: float = 0.35,
    sc: Optional[SuperCompress] = None,
    min_words: int = 100,
    keep_last_turns: int = 4,
):
    """
    Build a LangGraph node function: `state -> state_update`. Drop it into a
    graph right before your LLM-calling node.

        from supercompress_local.langgraph_adapter import make_compression_node

        graph.add_node("compress", make_compression_node(budget_ratio=0.3, keep_last_turns=4))
        graph.add_edge("compress", "call_model")

    keep_last_turns: the most recent N messages are always passed through
    unmodified — raise this if you want more of the recent back-and-forth
    guaranteed intact, lower it (min 1) to compress more aggressively.
    """
    engine = sc or SuperCompress()

    def _node(state: dict) -> dict:
        messages = state.get("messages", [])
        result = compress_messages(
            messages,
            budget_ratio=budget_ratio,
            sc=engine,
            min_words=min_words,
            keep_last_turns=keep_last_turns,
        )
        return {"messages": result["messages"]}

    return _node


__all__ = ["assemble_messages", "compress_messages", "make_compression_node"]
