"""
Add embeddings to each chunk in relations_ready.json using OpenAI API.
Loads OPENAI_API_KEY from .env. Writes updated relations_ready.json in place.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _PROJECT_ROOT / ".env"
_JSON_PATH = _PROJECT_ROOT / "data" / "processed" / "relations_ready.json"
_BATCH_SIZE = 100  # chunks per API call
_EMBEDDING_MODEL = "text-embedding-3-small"


def collect_chunks(data: dict) -> list[tuple[list, int, int]]:
    """
    Yield (chunks_list, section_idx, chunk_idx) for each chunk.
    chunks_list is the actual list so we can mutate in place.
    """
    for block_key in ("eu_ai_act", "korea_ai_law"):
        block = data.get(block_key) or {}
        for section in ("rationale", "articles", "annexes"):
            items = block.get(section) or []
            for si, item in enumerate(items):
                chunks = item.get("chunks") or []
                for ci in range(len(chunks)):
                    yield (chunks, si, ci)


def main() -> None:
    load_dotenv(_ENV_PATH)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not found in .env")

    with open(_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    refs: list[tuple[list, int]] = []
    texts: list[str] = []
    for chunks_list, si, ci in collect_chunks(data):
        chunk = chunks_list[ci]
        text = chunk.get("text") or ""
        if not text.strip():
            continue
        refs.append((chunks_list, ci))
        texts.append(text)

    client = OpenAI(api_key=api_key)
    total = len(texts)
    print(f"Embedding {total} chunks in batches of {_BATCH_SIZE}...")

    for i in range(0, total, _BATCH_SIZE):
        batch_texts = texts[i : i + _BATCH_SIZE]
        batch_refs = refs[i : i + _BATCH_SIZE]
        resp = client.embeddings.create(model=_EMBEDDING_MODEL, input=batch_texts)
        for j, emb_obj in enumerate(resp.data):
            if j < len(batch_refs):
                chunks_list, ci = batch_refs[j]
                chunks_list[ci]["embedding"] = emb_obj.embedding
        print(f"  {min(i + _BATCH_SIZE, total)}/{total}")

    with open(_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Wrote {_JSON_PATH} with embeddings.")


if __name__ == "__main__":
    main()
