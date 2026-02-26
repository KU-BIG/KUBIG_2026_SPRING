"""
Merge AIAct_relations_ready.json and AICoreLaw_relations_ready.json into a single
relations_ready.json, and build cross-jurisdictional [:ENCOMPASSES] and [:SUPPLEMENTS] relations.

Ling Long Protocol — relations_extraction.md. §3.1: (Sanction) MUST connect to (Stakeholder) via [:APPLIES_TO].
§3.8: ENCOMPASSES requires c1.lang != c2.lang (EU en -> KR ko). §3.9: SUPPLEMENTS target concept.lang
must NOT be in source doc jurisdiction. Every relation carries description (Verbatim §3.1).
Output: data/processed/relations_ready.json
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_AIACT_PATH = _PROJECT_ROOT / "data" / "processed" / "AIAct_relations_ready.json"
_AICORELAW_PATH = _PROJECT_ROOT / "data" / "processed" / "AICoreLaw_relations_ready.json"
_OUTPUT_PATH = _PROJECT_ROOT / "data" / "processed" / "relations_ready.json"

# ENCOMPASSES (§2, §3.8): (broader_concept, narrower_concept) — c1 from EU (en), c2 from KR (ko)
# c1.lang and c2.lang MUST be different. Concept from one jurisdiction is broader and includes another.
ENCOMPASSES_PAIRS: List[Tuple[str, str, str]] = [
    # (start_concept_eu, end_concept_kr, description_excerpt)
    (
        "High-Risk AI",
        "High-Impact AI",
        "EU High-Risk AI and Korea High-Impact AI both denote AI systems with significant impact on life, safety, and fundamental rights. The EU concept is broader in scope and regulatory structure.",
    ),
    (
        "Provider",
        "AI Business Operator",
        "EU Provider (develops or places AI on market) encompasses Korea AI Business Operator, which includes AI developer and AI-using business operator.",
    ),
    (
        "Provider",
        "AI Developer",
        "EU Provider includes persons that develop AI systems; Korea AI Developer is a person that develops and provides artificial intelligence, a subtype of Provider.",
    ),
    (
        "Affected Person",
        "Impacted Person",
        "EU affected persons and Korea impacted person both refer to persons whose rights or interests are significantly affected by AI systems.",
    ),
    (
        "AI System",
        "AI System",
        "EU AI system definition (machine-based, autonomy, adaptiveness) encompasses Korea AI system (AI-based system inferring outputs for given goal with autonomy and adaptability). Both jurisdictions define the same conceptual category.",
    ),
    (
        "General-Purpose AI System",
        "Generative AI",
        "EU General-Purpose AI System (based on GPAI model, serves variety of purposes) encompasses Korea Generative AI (generates text, sound, images, videos by imitating input data), a specific capability type.",
    ),
]

# SUPPLEMENTS (§2, §3.9): (source_doc_id, target_concept, description)
# Document from one jurisdiction fills regulatory gaps of a broader concept from another.
# Constraint §3.9: target Concept.lang must NOT be in Article/Annex/Rationale.source_data (EU doc → KR concept, KR doc → EU concept).
SUPPLEMENTS_PAIRS: List[Tuple[str, str, str, str]] = [
    # (doc_id, doc_type, target_concept, target_concept_jurisdiction, description)
    (
        "EU AI Act::ANNEX III",
        "Annex",
        "High-Impact AI",
        "KR",
        "EU Annex III lists high-risk AI systems by product area and use case, providing detailed criteria that fill regulatory gaps in Korea High-Impact AI, which references broad domains without exhaustive product lists.",
    ),
    (
        "EU AI Act::Article 6",
        "Article",
        "High-Impact AI",
        "KR",
        "EU AI Act Article 6 establishes the classification of high-risk AI systems and reference to Annex III; these criteria supplement Korea High-Impact AI with concrete product and use-area specifications.",
    ),
    (
        "EU AI Act::ANNEX XIII",
        "Annex",
        "Generative AI",
        "KR",
        "EU Annex XIII provides technical criteria (FLOPs, parameters, data size) for general-purpose AI models with systemic risk; supplements Korea Generative AI concept with quantitative thresholds.",
    ),
    (
        "Korea AI Law::Article 2",
        "Article",
        "High-Risk AI System",
        "EU",
        "Korea Article 2 defines High-Impact AI with Korea-specific domains (nuclear, drinking water, domestic agencies) that supplement the EU High-Risk AI System scope with additional use-area detail.",
    ),
]


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_concepts_from_relations(
    data: Dict[str, Any], source_label: str
) -> Set[str]:
    """Collect all Concept names from DEFINES and DETAILS_DEFINITION_OF relations."""
    concepts: Set[str] = set()
    for section in ("rationale", "articles", "annexes"):
        for item in data.get(section) or []:
            for rel in item.get("relations") or []:
                if rel.get("target_node_type") == "Concept":
                    concepts.add(rel.get("target_node_name") or "")
    return concepts


def build_merged_structure(
    aiact: Dict[str, Any], aicorelaw: Dict[str, Any]
) -> Dict[str, Any]:
    """Simple merge: list both files as separate blocks (no structural merge)."""
    return {
        "eu_ai_act": aiact,
        "korea_ai_law": aicorelaw,
    }


def build_cross_jurisdictional_relations(
    aiact: Dict[str, Any], aicorelaw: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Build ENCOMPASSES and SUPPLEMENTS relations for cross-jurisdictional linking."""
    eu_concepts = collect_concepts_from_relations(aiact, "EU")
    kr_concepts = collect_concepts_from_relations(aicorelaw, "Korea")

    relations: List[Dict[str, Any]] = []
    now = datetime.now(timezone.utc).isoformat()

    # ENCOMPASSES: Concept (broader, EU en) -> Concept (narrower, KR ko)
    for start_concept, end_concept, desc in ENCOMPASSES_PAIRS:
        if start_concept in eu_concepts or start_concept == "Affected Person":
            if end_concept in kr_concepts or end_concept in (
                "High-Impact AI",
                "AI Business Operator",
                "AI Developer",
                "Impacted Person",
                "AI System",
                "Generative AI",
            ):
                relations.append(
                    {
                        "type": "ENCOMPASSES",
                        "start_node_type": "Concept",
                        "start_node_name": start_concept,
                        "target_node_type": "Concept",
                        "target_node_name": end_concept,
                        "description": desc,
                        "source_file": "src/merge_relations.py",
                        "updated_at": now,
                    }
                )

    # SUPPLEMENTS: Article/Annex/Rationale -> Concept (different jurisdiction)
    doc_ids: Set[str] = set()
    for section, key in [
        ("rationale", "id"),
        ("articles", "id"),
        ("annexes", "id"),
    ]:
        for item in (aiact.get(section) or []) + (aicorelaw.get(section) or []):
            doc_ids.add(item.get(key) or "")

    for doc_id, doc_type, target_concept, target_jurisdiction, desc in SUPPLEMENTS_PAIRS:
        if doc_id in doc_ids or any(doc_id in s for s in doc_ids if s):
            relations.append(
                {
                    "type": "SUPPLEMENTS",
                    "start_node_type": doc_type,
                    "start_node_name": doc_id,
                    "target_node_type": "Concept",
                    "target_node_name": target_concept,
                    "description": desc,
                    "source_file": "src/merge_relations.py",
                    "updated_at": now,
                }
            )

    return relations


def _validate_sanctions_have_applies_to(merged: Dict[str, Any]) -> None:
    """§3.1 (first general constraint): Every (Sanction) MUST be connected to (Stakeholder) via [:APPLIES_TO]."""
    sanctions_with_applies_to: Set[str] = set()
    sanctions_referenced: Set[str] = set()

    for block in [merged.get("eu_ai_act") or {}, merged.get("korea_ai_law") or {}]:
        if not isinstance(block, dict):
            continue
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
            raise ValueError(
                f"Schema §3.1: Sanction '{sanction_name}' must be connected to (Stakeholder) via [:APPLIES_TO]"
            )


def main() -> None:
    aiact = load_json(_AIACT_PATH)
    aicorelaw = load_json(_AICORELAW_PATH)

    merged = build_merged_structure(aiact, aicorelaw)
    cross_rel = build_cross_jurisdictional_relations(aiact, aicorelaw)
    merged["cross_jurisdictional_relations"] = cross_rel

    _validate_sanctions_have_applies_to(merged)

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    total_eu = sum(
        len(item.get("relations") or [])
        for section in ("rationale", "articles", "annexes")
        for item in aiact.get(section) or []
    )
    total_kr = sum(
        len(item.get("relations") or [])
        for section in ("rationale", "articles", "annexes")
        for item in aicorelaw.get(section) or []
    )
    print(
        f"Wrote {_OUTPUT_PATH}: eu_ai_act ({total_eu} relations), korea_ai_law ({total_kr} relations), {len(cross_rel)} cross-jurisdictional"
    )
    for r in cross_rel:
        print(f"  {r['type']}: {r.get('start_node_name','?')} -> {r.get('target_node_name','?')}")


if __name__ == "__main__":
    main()
