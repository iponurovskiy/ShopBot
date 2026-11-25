#!/usr/bin/env python3
"""
Dump the Crescendo escalation chain from PyRIT's SQLite memory.

    python3 dump_transcript.py              # the most recent attack result
    python3 dump_transcript.py <attack_result_id>

"""

import asyncio
import inspect
import sys

from pyrit.setup import initialize_pyrit_async
from pyrit.memory import CentralMemory


def sig(obj, name: str) -> str:
    fn = getattr(obj, name, None)
    if fn is None:
        return f"{name}: <missing>"
    try:
        return f"{name}{inspect.signature(fn)}"
    except (TypeError, ValueError):
        return f"{name}: <no signature>"


def attrs(obj) -> dict:
    out = {}
    for n in dir(obj):
        if n.startswith("_"):
            continue
        try:
            v = getattr(obj, n)
        except Exception: 
            continue
        if not callable(v):
            out[n] = v
    return out


def show_piece(p, idx: int) -> None:
    role = getattr(p, "role", "?")
    text = getattr(p, "converted_value", None) or getattr(p, "original_value", "") or ""
    text = str(text).replace("\n", " ")
    print(f"\n[{idx:>2}] {role.upper()}")
    print(f"     {text[:700]}")


async def main() -> None:
    await initialize_pyrit_async("SQLite")
    mem = CentralMemory.get_memory_instance()

    print("--- signatures ---")
    for name in ("get_attack_results", "get_conversation_messages", "get_message_pieces",
                 "get_scores", "get_prompt_scores"):
        print(f"  {sig(mem, name)}")

    target_id = sys.argv[1] if len(sys.argv) > 1 else None

    try:
        results = mem.get_attack_results()
    except TypeError as e:
        print(f"\nget_attack_results needs arguments: {e}")
        return

    if not results:
        print("\nno attack results in memory — was the run made with SQLite memory?")
        return

    print(f"\n--- {len(results)} attack result(s) in memory ---")
    result = results[-1]
    if target_id:
        for r in results:
            if str(getattr(r, "id", "")) == target_id or str(
                getattr(r, "attack_result_id", "")
            ) == target_id:
                result = r
                break

    print("\n--- chosen result ---")
    for k, v in attrs(result).items():
        s = str(v)
        print(f"  {k}: {s[:160]}")

    # The conversation id is what ties the result to its message pieces.
    conv_id = (
        getattr(result, "conversation_id", None)
        or getattr(result, "last_response", None)
        and getattr(result.last_response, "conversation_id", None)
    )
    print(f"\nconversation_id: {conv_id}")

    if not conv_id:
        print("no conversation_id on the result — check the attrs above for the right field")
        return

    print("\n=== escalation chain ===")
    pieces = None
    for getter, kwargs in (
        ("get_conversation_messages", {"conversation_id": conv_id}),
        ("get_message_pieces", {"conversation_id": conv_id}),
    ):
        try:
            pieces = getattr(mem, getter)(**kwargs)
            print(f"(via {getter})")
            break
        except Exception as e: 
            print(f"  {getter} failed: {type(e).__name__}: {e}")

    if not pieces:
        print("could not fetch the conversation — see the signatures above")
        return

    idx = 0
    for item in pieces:
        inner = getattr(item, "message_pieces", None) or getattr(item, "request_pieces", None)
        for p in (inner if inner else [item]):
            idx += 1
            show_piece(p, idx)

    # Scores explain why each turn was accepted, refused, or backtracked.
    print("\n=== scores ===")
    try:
        scores = mem.get_scores()
        for s in scores[-20:]:
            print(f"  {getattr(s, 'score_type', '?')} "
                  f"{getattr(s, 'score_category', '')} = {getattr(s, 'score_value', '?')} "
                  f"| {str(getattr(s, 'score_rationale', ''))[:220]}")
    except Exception as e:  
        print(f"  get_scores failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
