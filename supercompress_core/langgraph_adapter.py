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


def compress_messages(
    messages: list,
    budget_ratio: float = 0.35,
    sc: Optional[SuperCompress] = None,
    min_words: int = 100,
) -> dict:
    """
    Compress a LangGraph-style message list. Returns a dict:
        {"messages": [...], "original_tokens": int, "kept_tokens": int,
         "tokens_saved": int, "savings_pct": float, "skipped": bool}

    The returned `messages` list has the same shape as the input (list of
    dicts) with the history collapsed into one system-role summary message,
    the original system prompt (if any) kept first, and the last user
    message passed through untouched.
    """
    context, query, system_msg = assemble_messages(messages)

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

    engine = sc or SuperCompress()
    result = engine.compress(context, query=query, budget_ratio=budget_ratio)

    out = []
    if system_msg is not None:
        out.append(
            system_msg
            if isinstance(system_msg, dict)
            else {"role": "system", "content": _content_of(system_msg)}
        )
    out.append(
        {
            "role": "system",
            "content": f"[Compressed context — {result.tokens_saved} tokens saved (~{round(result.kv_savings_pct)}%)]\n\n{result.compressed_text}",
        }
    )
    last_user = next((m for m in reversed(messages) if _role_of(m) == "user"), None)
    if last_user is not None:
        out.append(
            last_user
            if isinstance(last_user, dict)
            else {"role": "user", "content": _content_of(last_user)}
        )

    return {
        "messages": out,
        "original_tokens": result.original_tokens,
        "kept_tokens": result.kept_tokens,
        "tokens_saved": result.tokens_saved,
        "savings_pct": round(result.kv_savings_pct, 1),
        "skipped": False,
    }


def make_compression_node(
    budget_ratio: float = 0.35, sc: Optional[SuperCompress] = None, min_words: int = 100
):
    """
    Build a LangGraph node function: `state -> state_update`. Drop it into a
    graph right before your LLM-calling node.

        from supercompress_local.langgraph_adapter import make_compression_node

        graph.add_node("compress", make_compression_node(budget_ratio=0.3))
        graph.add_edge("compress", "call_model")
    """
    engine = sc or SuperCompress()

    def _node(state: dict) -> dict:
        messages = state.get("messages", [])
        result = compress_messages(
            messages, budget_ratio=budget_ratio, sc=engine, min_words=min_words
        )
        return {"messages": result["messages"]}

    return _node


__all__ = ["assemble_messages", "compress_messages", "make_compression_node"]
