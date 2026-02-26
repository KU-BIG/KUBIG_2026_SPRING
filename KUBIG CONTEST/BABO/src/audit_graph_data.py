"""
Verify *_relations_ready.json against the Ling Long Protocol schema (relations_extraction.md).
Ensures every relation's Start Node and End Node match the schema STRICTLY.
Validates Verbatim (§3.1), Relevance (§3.6), and §3.1 first general constraint: (Sanction) MUST connect to (Stakeholder) via [:APPLIES_TO].
Node Centered (§3.4) and Node Integrity (§3.5) are enforced at extraction and upload.

Usage: python src/audit_graph_data.py [path/to/file.json ...]
        Default: data/processed/AIAct_relations_ready.json data/processed/AICoreLaw_relations_ready.json
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_INPUTS = [
    _PROJECT_ROOT / "data" / "processed" / "relations_ready.json",
    _PROJECT_ROOT / "data" / "processed" / "AIAct_relations_ready.json",
    _PROJECT_ROOT / "data" / "processed" / "AICoreLaw_relations_ready.json",
]

# Schema §2: Relationship Extraction Logic — Start Node and End Node (exact)
# Start/End are the node TYPES that may appear for that relation.
RELATION_SCHEMA: Dict[str, Dict[str, List[str]]] = {
    "MANDATED_FOR": {"start": ["Requirement"], "end": ["Stakeholder"]},
    "PERMITS": {"start": ["Requirement"], "end": ["Stakeholder"]},
    "DEFINES": {"start": ["Article"], "end": ["Concept"]},
    "DETAILS_DEFINITION_OF": {"start": ["Rationale", "Annex"], "end": ["Concept"]},
    "DETAILS_SCOPE": {"start": ["Rationale", "Annex", "Article"], "end": ["Domain"]},  # Article exempt for KR/EU definition articles
    "JUSTIFIES_LOGIC": {"start": ["Rationale", "Annex"], "end": ["Article"]},
    "INCLUDES": {"start": ["Regulation"], "end": ["Article", "Rationale", "Annex"]},
    "ESTABLISHES": {"start": ["Article", "Rationale", "Annex"], "end": ["TechCriterion", "UsageCriterion"]},
    "TRIGGERS": {"start": ["Domain"], "end": ["Requirement", "Sanction"]},
    "LEADS_TO": {"start": ["Domain", "TechCriterion", "UsageCriterion"], "end": ["RiskCategory"]},
    "IMPOSES": {"start": ["Article"], "end": ["Requirement"]},
    "PENALIZES_WITH": {"start": ["Article"], "end": ["Sanction"]},
    "IS_A": {"start": ["Article"], "end": ["Support"]},
    # Schema §2: APPLIES_TO — Sanction -> Stakeholder (people/body responsible for the sanction)
    "APPLIES_TO": {"start": ["Sanction"], "end": ["Stakeholder"]},
    # Cross-jurisdictional (Ling Long Protocol §2)
    "ENCOMPASSES": {"start": ["Concept"], "end": ["Concept"]},
    "SUPPLEMENTS": {"start": ["Article", "Annex", "Rationale"], "end": ["Concept"]},
}

# Relations that are stored under Article but logically start from Requirement (chain)
# We accept Article as container and only validate End node.
CHAIN_RELATIONS_STORED_UNDER_ARTICLE = {"MANDATED_FOR", "PERMITS"}


def get_container_type(container_id: str, section: str) -> str:
    """Map section key to schema node type."""
    if section == "rationale":
        return "Rationale"
    if section == "articles":
        return "Article"
    if section == "annexes":
        return "Annex"
    if "Recital" in container_id or "recital" in container_id.lower():
        return "Rationale"
    if "ANNEX" in container_id.upper() or "Annex" in container_id:
        return "Annex"
    return "Article"


def get_effective_start_type(rel: Dict[str, Any], container_type: str) -> Optional[str]:
    """Return the schema Start Node type for this relation."""
    rel_type = rel.get("type")
    if not rel_type:
        return None
    # Explicit start (e.g. TRIGGERS, LEADS_TO with start_node_type)
    if rel.get("start_node_type"):
        return rel["start_node_type"]
    # Chain relations: logical start is Requirement; we accept Article as container
    if rel_type in CHAIN_RELATIONS_STORED_UNDER_ARTICLE:
        return "Requirement"  # Logical start; container is Article
    return container_type


def validate_relation(
    rel: Dict[str, Any],
    container_id: str,
    container_type: str,
    index: int,
    file_path: Path,
) -> List[str]:
    """Validate one relation. Returns list of error messages (empty if valid)."""
    errors: List[str] = []
    rel_type = rel.get("type")
    target_type = rel.get("target_node_type")
    desc = rel.get("description")

    if not rel_type:
        errors.append(f"[{file_path}] {container_id} relation[{index}]: missing 'type'")
        return errors

    if rel_type not in RELATION_SCHEMA:
        errors.append(
            f"[{file_path}] {container_id} relation[{index}]: unknown relation type '{rel_type}'"
        )
        return errors

    rule = RELATION_SCHEMA[rel_type]
    start_ok = rule["start"]
    end_ok = rule["end"]

    # 1. Verbatim (§3.1): every relation MUST have description; (Sanction) MUST connect to (Stakeholder) via [:APPLIES_TO]
    if not desc or not str(desc).strip():
        errors.append(
            f"[{file_path}] {container_id} relation[{index}] type={rel_type}: missing or empty 'description' (schema §3.1)"
        )
    # Relevance (§3.6): description should be directly relevant excerpt, not full-text
    elif len(str(desc).strip()) > 3000:
        errors.append(
            f"[{file_path}] {container_id} relation[{index}] type={rel_type}: 'description' excessively long "
            "(schema §3.6 Relevance — extract only directly relevant parts, not full-text)"
        )

    # 2. End Node (target_node_type) MUST match schema
    if not target_type:
        errors.append(
            f"[{file_path}] {container_id} relation[{index}] type={rel_type}: missing 'target_node_type'"
        )
    elif target_type not in end_ok:
        errors.append(
            f"[{file_path}] {container_id} relation[{index}] type={rel_type}: End Node must be one of {end_ok}, got '{target_type}'"
        )

    # 3. Start Node: either explicit start_node_type or container
    effective_start = get_effective_start_type(rel, container_type)
    if effective_start and effective_start not in start_ok:
        errors.append(
            f"[{file_path}] {container_id} relation[{index}] type={rel_type}: Start Node must be one of {start_ok}, got '{effective_start}'"
        )

    # 4. When relation has explicit start_node_type, it must match schema start
    if rel.get("start_node_type") and rel["start_node_type"] not in start_ok:
        errors.append(
            f"[{file_path}] {container_id} relation[{index}] type={rel_type}: start_node_type must be one of {start_ok}, got '{rel['start_node_type']}'"
        )

    return errors


def verify_file(file_path: Path) -> Tuple[int, int, List[str]]:
    """
    Verify one *_relations_ready.json file.
    Returns (total_relations, error_count, list of error messages).
    """
    all_errors: List[str] = []
    total_relations = 0

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return 0, 1, [f"[{file_path}] Failed to load JSON: {e}"]

    # Simple-merge format: eu_ai_act + korea_ai_law blocks
    if "eu_ai_act" in data and "korea_ai_law" in data:
        for block_name, block in [("eu_ai_act", data["eu_ai_act"]), ("korea_ai_law", data["korea_ai_law"])]:
            for section in ("rationale", "articles", "annexes"):
                items = block.get(section) or []
                for item in items:
                    container_id = item.get("id", "")
                    relations = item.get("relations") or []
                    for i, rel in enumerate(relations):
                        total_relations += 1
                        container_type = get_container_type(container_id, section)
                        errs = validate_relation(
                            rel, f"{block_name}::{container_id}", container_type, i, file_path
                        )
                        all_errors.extend(errs)
    else:
        # Standalone format: top-level rationale, articles, annexes
        for section in ("rationale", "articles", "annexes"):
            items = data.get(section) or []
            for item in items:
                container_id = item.get("id", "")
                relations = item.get("relations") or []
                for i, rel in enumerate(relations):
                    total_relations += 1
                    container_type = get_container_type(container_id, section)
                    errs = validate_relation(
                        rel, container_id, container_type, i, file_path
                    )
                    all_errors.extend(errs)

    # Cross-jurisdictional relations (standalone, explicit start/end)
    for i, rel in enumerate(data.get("cross_jurisdictional_relations") or []):
        total_relations += 1
        container_id = rel.get("start_node_name") or f"cross_rel_{i}"
        container_type = rel.get("start_node_type") or "Concept"
        errs = validate_relation(rel, container_id, container_type, i, file_path)
        all_errors.extend(errs)

    # §3.1 (first general constraint): Every (Sanction) MUST be connected to (Stakeholder) via [:APPLIES_TO]
    sanctions_with_applies_to, sanctions_referenced = set(), set()
    for _block_name, block in _iter_blocks(data):
        for item in (block.get("rationale") or []) + (block.get("articles") or []) + (block.get("annexes") or []):
            for rel in item.get("relations") or []:
                rtype = rel.get("type") or ""
                target_type = rel.get("target_node_type") or ""
                if rtype == "PENALIZES_WITH" or (rtype == "TRIGGERS" and target_type == "Sanction"):
                    sanctions_referenced.add(rel.get("target_node_name") or "")
                elif rtype == "APPLIES_TO" and rel.get("start_node_type") == "Sanction":
                    sanctions_with_applies_to.add(rel.get("start_node_name") or "")
    for sanction_name in sanctions_referenced:
        if sanction_name and sanction_name not in sanctions_with_applies_to:
            all_errors.append(
                f"[{file_path}] Sanction '{sanction_name}': MUST be connected to (Stakeholder) via [:APPLIES_TO] (schema §3.1 first general constraint)"
            )

    return total_relations, len(all_errors), all_errors


def _iter_blocks(data: dict):
    """Yield (block_name, block) for eu_ai_act, korea_ai_law, or standalone sections."""
    if "eu_ai_act" in data and "korea_ai_law" in data:
        yield "eu_ai_act", data["eu_ai_act"] or {}
        yield "korea_ai_law", data["korea_ai_law"] or {}
    else:
        yield "data", data


def main() -> None:
    if len(sys.argv) > 1:
        paths = [Path(p) for p in sys.argv[1:]]
    else:
        paths = _DEFAULT_INPUTS

    total_rel = 0
    total_err = 0
    for path in paths:
        if not path.is_absolute():
            path = _PROJECT_ROOT / path
        if not path.exists():
            print(f"Skip (not found): {path}")
            continue
        n_rel, n_err, errors = verify_file(path)
        total_rel += n_rel
        total_err += n_err
        status = "PASS" if n_err == 0 else "FAIL"
        print(f"{path.name}: {n_rel} relations, {n_err} errors — {status}")
        for e in errors:
            print(f"  {e}")

    print("---")
    if total_err == 0:
        print("Overall: PASS — All relations match schema (Start and End nodes).")
    else:
        print(f"Overall: FAIL — {total_err} validation error(s) across {total_rel} relations.")
    sys.exit(1 if total_err > 0 else 0)


if __name__ == "__main__":
    main()
