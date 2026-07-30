"""
Data loaders for common conversation formats:
- WildChat (from Hugging Face datasets)
- ShareGPT format
- OpenAI chat format
- Generic JSONL
"""

import json
from typing import Dict, Iterable, List, Optional


def load_conversations(
    path: str,
    format: str = "sharegpt",
    n: Optional[int] = None,
) -> List[Dict]:
    """
    Load conversations from a file.

    Args:
        path: Path to the file.
        format: One of "sharegpt", "openai", "jsonl".
            - sharegpt: [{"conversations": [{"from": ..., "value": ...}]}]
            - openai: [{"messages": [{"role": ..., "content": ...}]}]
            - jsonl: One JSON object per line, with a "conversation" or
              "conversations" field.
        n: Maximum number of conversations to load.

    Returns:
        List of conversations in sigx internal format.
    """
    if format == "sharegpt":
        return _load_sharegpt(path, n)
    elif format == "openai":
        return _load_openai(path, n)
    elif format == "jsonl":
        return _load_jsonl(path, n)
    else:
        raise ValueError(f"Unknown format: {format}")


def load_wildchat(
    path: Optional[str] = None,
    split: str = "train",
    n: Optional[int] = None,
) -> List[Dict]:
    """
    Load WildChat dataset from Hugging Face.

    Requires: pip install datasets

    Args:
        path: Local path to WildChat data (if already downloaded).
            If None, loads from Hugging Face hub.
        split: Dataset split (default "train").
        n: Maximum number of conversations to load.

    Returns:
        List of conversations in sigx internal format.
    """
    if path is not None:
        return load_conversations(path, format="jsonl", n=n)

    try:
        from datasets import load_dataset
    except ImportError as err:
        raise ImportError(
            "datasets library required. Install with: pip install datasets"
        ) from err

    ds = load_dataset("allenai/WildChat-1M", split=split, streaming=(n is None))
    if n is not None:
        ds = ds.take(n)

    return [_normalize_wildchat(item) for item in ds]


def _normalize_wildchat(item: Dict) -> Dict:
    """Normalize WildChat item to sigx format."""
    timestamp = item.get("timestamp", "")
    try:
        timestamp = str(timestamp)
    except Exception:
        timestamp = ""

    return {
        "conversation_id": item.get("conversation_hash", ""),
        "model": item.get("model", ""),
        "timestamp": timestamp,
        "conversation": [
            {"role": t.get("role", ""), "content": t.get("content", "")}
            for t in item.get("conversation", [])
        ],
    }


def _load_sharegpt(path: str, n: Optional[int] = None) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if n is not None:
        data = data[:n]

    result = []
    for i, item in enumerate(data):
        convs = item.get("conversations", [])
        result.append({
            "conversation_id": str(i),
            "conversation": [
                {
                    "role": "user" if c.get("from") == "human" else "assistant",
                    "content": c.get("value", ""),
                }
                for c in convs
            ],
        })

    return result


def _load_openai(path: str, n: Optional[int] = None) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if n is not None:
        data = data[:n]

    result = []
    for i, item in enumerate(data):
        messages = item.get("messages", [])
        result.append({
            "conversation_id": str(i),
            "conversation": [
                {"role": m.get("role", "user"), "content": m.get("content", "")}
                for m in messages
            ],
        })

    return result


def _load_jsonl(path: str, n: Optional[int] = None) -> List[Dict]:
    result = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if n is not None and i >= n:
                break
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            conv = item.get("conversation") or item.get("conversations")
            if conv is None:
                continue
            result.append({
                "conversation_id": item.get(
                    "conversation_id", item.get("conversation_hash", str(i))
                ),
                "conversation": [
                    {
                        "role": t.get("role", t.get("from", "user")),
                        "content": t.get("content", t.get("value", "")),
                    }
                    for t in conv
                ],
            })

    return result


def stream_wildchat(n: Optional[int] = None) -> Iterable[Dict]:
    """Stream WildChat from Hugging Face (generator)."""
    try:
        from datasets import load_dataset
    except ImportError as err:
        raise ImportError(
            "datasets library required. Install with: pip install datasets"
        ) from err

    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    count = 0
    for item in ds:
        yield _normalize_wildchat(item)
        count += 1
        if n is not None and count >= n:
            break
