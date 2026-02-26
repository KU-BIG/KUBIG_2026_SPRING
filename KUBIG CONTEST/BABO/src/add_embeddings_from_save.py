"""
Add embeddings to each chunk in relations_ready.json by copying from save_chunk_ready.json.

Match chunks by (block, section, item_id, chunk_id). Fast, no API cost.
Writes updated relations_ready.json in place.
"""

import json
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RELATIONS_PATH = _PROJECT_ROOT / "data" / "processed" / "relations_ready.json"
_SAVE_CHUNK_PATH = _PROJECT_ROOT / "data" / "save_chunk_ready.json"


def build_embedding_index(save_data: dict) -> dict:
    """Build (block_key, item_id, chunk_id) -> embedding."""
    index: dict = {}
    for block_key in ("eu_ai_act", "korea_ai_law"):
        block = save_data.get(block_key) or {}
        for section in ("rationale", "articles", "annexes"):
            for item in block.get(section) or []:
                item_id = item.get("id") or ""
                for chunk in item.get("chunks") or []:
                    chunk_id = chunk.get("chunk_id") or ""
                    emb = chunk.get("embedding")
                    if item_id and chunk_id and emb is not None:
                        index[(block_key, item_id, chunk_id)] = emb
    return index


def main() -> None:
    with open(_RELATIONS_PATH, "r", encoding="utf-8") as f:
        relations = json.load(f)
    with open(_SAVE_CHUNK_PATH, "r", encoding="utf-8") as f:
        save_data = json.load(f)

    index = build_embedding_index(save_data)
    matched = 0
    missed = 0

    for block_key in ("eu_ai_act", "korea_ai_law"):
        block = relations.get(block_key) or {}
        for section in ("rationale", "articles", "annexes"):
            for item in block.get(section) or []:
                item_id = item.get("id") or ""
                for chunk in item.get("chunks") or []:
                    chunk_id = chunk.get("chunk_id") or ""
                    key = (block_key, item_id, chunk_id)
                    if key in index:
                        chunk["embedding"] = index[key]
                        matched += 1
                    else:
                        missed += 1

    with open(_RELATIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(relations, f, indent=2, ensure_ascii=False)

    print(f"Wrote {_RELATIONS_PATH}: {matched} chunks with embeddings, {missed} missed (no match in save_chunk_ready).")


if __name__ == "__main__":
    main()
