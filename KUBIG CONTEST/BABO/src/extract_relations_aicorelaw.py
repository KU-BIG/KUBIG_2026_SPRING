"""
Extract nodes and relations from AICoreLaw_graph_chunk.json following the Ling Long Protocol
(cursor/rules/relations_extraction.md) for Korea AI Law.

Output: AICoreLaw_relations_ready.json with embedded relations per article.
Schema: Ling Long Protocol. §3.1: (Sanction) MUST connect to (Stakeholder) via [:APPLIES_TO].
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SOURCE_FILE = "AICoreLaw_graph_chunk.json"
_OUTPUT_FILE = "AICoreLaw_relations_ready.json"
_REGULATION_NAME = "Korea AI Law"
_EFFECTIVE_DATE = "2025-03-14"  # 1 year after promulgation per Addenda
_SOURCE_FILE_PATH = "data/processed/AICoreLaw_graph_chunk.json"

# KR-specific: Domains from Article 2(4) high-impact AI areas
DOMAINS_KR = [
    ("Critical Infrastructure", "Energy, water, nuclear - supply, production, management"),
    ("Healthcare", "Health and medical services, medical devices"),
    ("Law Enforcement", "Biometric information for criminal investigation or arrests"),
    ("Employment", "Hiring, loan screening - rights and obligations"),
    ("Transportation", "Means of transportation, traffic facilities and systems"),
    ("Administration of Justice", "State agencies decision-making, public services"),
    ("Education", "Student evaluation in early childhood, elementary, secondary education"),
]

# KR stakeholders (schema: use EXACT same string as Concept where applicable)
KR_STAKEHOLDERS = [
    "AI Business Operator",
    "AI Developer",
    "User",
    "Impacted Person",
    "Domestic Agent",
]


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _node_id(regulation: str, raw_id: str) -> str:
    """Article.id format: jurisdiction + law + Article number."""
    return f"{regulation}::{raw_id}"


def _to_title_concept_kr(term: str) -> str:
    """Normalize KR term to Concept title form (no source tags). Schema: Title Form."""
    lower = term.lower().strip()
    mapping = {
        "artificial intelligence": "Artificial Intelligence",
        "artificial intelligence system": "AI System",
        "artificial intelligence technology": "AI Technology",
        "high-impact artificial intelligence": "High-Impact AI",
        "generative artificial intelligence": "Generative AI",
        "artificial intelligence industry": "AI Industry",
        "artificial intelligence business operator": "AI Business Operator",
        "artificial intelligence developer": "AI Developer",
        "user": "User",
        "impacted person": "Impacted Person",
        "artificial intelligence society": "AI Society",
        "artificial intelligence ethics": "AI Ethics",
        "domestic agent": "Domestic Agent",
    }
    return mapping.get(lower) or term.title()


def extract_article2_defines(article: dict) -> List[dict]:
    """(Article)-[:DEFINES]->(Concept) for Article 2 Definitions. KR format."""
    relations = []
    full_text = article.get("full_text", "")
    if "Definitions" not in full_text or "means" not in full_text.lower():
        return relations

    # KR format: 1. The term "X" means ... or 7. (a) Artificial intelligence developer: ...
    pattern = r'\d+\.\s+The term\s+"([^"]+)"\s+means\s+'
    for m in re.finditer(pattern, full_text):
        term = m.group(1).strip()
        name = _to_title_concept_kr(term)
        desc_start = m.end()
        next_def = re.search(r';\s*\n\d+\.', full_text[desc_start:])
        if next_def:
            excerpt = full_text[desc_start : desc_start + next_def.start() + 1].strip()
        else:
            excerpt = full_text[desc_start : desc_start + 600].strip()
        if excerpt and len(excerpt) > 15:
            node_type = "Stakeholder" if name in KR_STAKEHOLDERS else "Concept"
            relations.append({
                "type": "DEFINES",
                "target_node_type": node_type,
                "target_node_name": name,
                "description": excerpt[:800],
            })
    # Sub-definition (a): Artificial intelligence developer
    sub_m = re.search(r'\(a\)\s+Artificial intelligence developer:\s+([^;]+)', full_text, re.I)
    if sub_m:
        relations.append({
            "type": "DEFINES",
            "target_node_type": "Stakeholder",
            "target_node_name": "AI Developer",
            "description": sub_m.group(0)[:800],
        })
    return relations


def extract_article2_establishes_leads_to(article: dict) -> List[dict]:
    """Article 2(4): (Article)-[:ESTABLISHES]->(UsageCriterion), Domain-[:LEADS_TO]->RiskCategory."""
    relations = []
    full_text = article.get("full_text", "")
    if "high-impact artificial intelligence" not in full_text.lower():
        return relations
    relations.append({
        "type": "ESTABLISHES",
        "target_node_type": "UsageCriterion",
        "target_node_name": "High-impact AI systems by use area",
        "description": full_text[:800],
    })
    domain_patterns = [
        ("Critical Infrastructure", r"energy|water|nuclear"),
        ("Healthcare", r"health|medical"),
        ("Law Enforcement", r"criminal|biometric|arrests"),
        ("Employment", r"hiring|loan"),
        ("Transportation", r"transportation|traffic"),
        ("Administration of Justice", r"State agencies|public services"),
        ("Education", r"education|students"),
    ]
    for domain_name, pat in domain_patterns:
        if re.search(pat, full_text, re.I):
            relations.append({
                "type": "LEADS_TO",
                "target_node_type": "RiskCategory",
                "target_node_name": "High-Impact AI",
                "start_node_type": "Domain",
                "start_node_name": domain_name,
                "description": full_text[:600],
            })
    return relations[:15]


def extract_article2_details_scope(article: dict) -> List[dict]:
    """Article 2(4): (Article)-[:DETAILS_SCOPE]->(Domain) for high-impact AI areas."""
    relations = []
    full_text = article.get("full_text", "")
    domain_keywords = [
        ("Critical Infrastructure", r"energy|water|nuclear|drinking water"),
        ("Healthcare", r"health|medical|medical devices"),
        ("Law Enforcement", r"criminal investigation|arrests|biometric"),
        ("Employment", r"hiring|loan screening"),
        ("Transportation", r"transportation|traffic"),
        ("Administration of Justice", r"State agencies|public services|qualifications"),
        ("Education", r"education|students|evaluation"),
    ]
    for domain_name, pat in domain_keywords:
        if re.search(pat, full_text, re.I) and "high-impact" in full_text.lower():
            relations.append({
                "type": "DETAILS_SCOPE",
                "target_node_type": "Domain",
                "target_node_name": domain_name,
                "description": full_text[:600],
            })
    return relations[:10]


def extract_imposes_mandated_for(article: dict) -> List[dict]:
    """(Article)-[:IMPOSES]->(Requirement)-[:MANDATED_FOR]->(Stakeholder)."""
    relations = []
    full_text = article.get("full_text", "")
    if not re.search(r"\b(shall|must|required to|obliged to)\b", full_text, re.I):
        return relations
    seen = set()
    if "artificial intelligence business operator" in full_text.lower() and "AI Business Operator" not in seen:
        relations.append({
            "type": "MANDATED_FOR",
            "target_node_type": "Stakeholder",
            "target_node_name": "AI Business Operator",
            "description": full_text[:800],
        })
        seen.add("AI Business Operator")
    if "impacted person" in full_text.lower() and "Impacted Person" not in seen:
        relations.append({
            "type": "MANDATED_FOR",
            "target_node_type": "Stakeholder",
            "target_node_name": "Impacted Person",
            "description": full_text[:800],
        })
        seen.add("Impacted Person")
    if "domestic agent" in full_text.lower() and "Domestic Agent" not in seen:
        relations.append({
            "type": "MANDATED_FOR",
            "target_node_type": "Stakeholder",
            "target_node_name": "Domestic Agent",
            "description": full_text[:800],
        })
        seen.add("Domestic Agent")
    return relations[:6]


def extract_permits(article: dict) -> List[dict]:
    """(Requirement)-[:PERMITS]->(Stakeholder). Trigger: may, allow, be entitled to."""
    relations = []
    full_text = article.get("full_text", "")
    if not re.search(r"\b(may|allow|be entitled to|entitled to)\b", full_text, re.I):
        return relations
    if "impacted person" in full_text.lower() and "entitled" in full_text.lower():
        relations.append({
            "type": "PERMITS",
            "target_node_type": "Stakeholder",
            "target_node_name": "Impacted Person",
            "description": full_text[:800],
        })
    if "artificial intelligence business operator" in full_text.lower():
        relations.append({
            "type": "PERMITS",
            "target_node_type": "Stakeholder",
            "target_node_name": "AI Business Operator",
            "description": full_text[:800],
        })
    return relations[:3]


def extract_article_imposes_requirement(article: dict) -> List[dict]:
    """(Article)-[:IMPOSES]->(Requirement)."""
    relations = []
    full_text = article.get("full_text", "")
    if not re.search(r"\b(shall|must|required to|obliged to)\b", full_text, re.I):
        return relations
    relations.append({
        "type": "IMPOSES",
        "target_node_type": "Requirement",
        "target_node_name": f"Obligation under {article.get('id', '')}",
        "description": full_text[:800],
    })
    return relations


def extract_sanctions(article: dict) -> List[dict]:
    """(Article)-[:PENALIZES_WITH]->(Sanction). §3.1: (Sanction) MUST connect to (Stakeholder) via [:APPLIES_TO]."""
    relations = []
    full_text = article.get("full_text", "")
    if not re.search(r"fine|penalty|punished|administrative fine|imprisonment|million won", full_text, re.I):
        return relations

    sanction_name = f"Sanction under {article.get('id', '')}"
    relations.append({
        "type": "PENALIZES_WITH",
        "target_node_type": "Sanction",
        "target_node_name": sanction_name,
        "description": full_text[:800],
    })

    # §3.1: (Sanction)-[:APPLIES_TO]->(Stakeholder) — who the sanction applies to
    stakeholder_patterns = [
        ("AI Business Operator", r"business\s+operator|ai\s+business\s+operator"),
        ("AI Developer", r"ai\s+developer|artificial\s+intelligence\s+developer"),
        ("User", r"\busers?\b"),
        ("Impacted Person", r"impacted\s+person"),
        ("Domestic Agent", r"domestic\s+agent"),
    ]
    for sh, pat in stakeholder_patterns:
        if re.search(pat, full_text, re.I):
            relations.append({
                "type": "APPLIES_TO",
                "start_node_type": "Sanction",
                "start_node_name": sanction_name,
                "target_node_type": "Stakeholder",
                "target_node_name": sh,
                "description": full_text[:800],
            })
    if not any(r.get("type") == "APPLIES_TO" for r in relations):
        # Fallback: AI Business Operator / AI Developer are most common for KR sanctions
        if re.search(r"business operator|ai business operator", full_text, re.I):
            relations.append({
                "type": "APPLIES_TO",
                "start_node_type": "Sanction",
                "start_node_name": sanction_name,
                "target_node_type": "Stakeholder",
                "target_node_name": "AI Business Operator",
                "description": full_text[:800],
            })
        elif re.search(r"developer|ai developer", full_text, re.I):
            relations.append({
                "type": "APPLIES_TO",
                "start_node_type": "Sanction",
                "start_node_name": sanction_name,
                "target_node_type": "Stakeholder",
                "target_node_name": "AI Developer",
                "description": full_text[:800],
            })
        else:
            # §3.1: No Stakeholder extractable from text — use default AI Business Operator
            relations.append({
                "type": "APPLIES_TO",
                "start_node_type": "Sanction",
                "start_node_name": sanction_name,
                "target_node_type": "Stakeholder",
                "target_node_name": "AI Business Operator",
                "description": full_text[:800] + " [Default Stakeholder: no Stakeholder could be extracted from the text; AI Business Operator assigned as default per §3.1.]",
                "is_default_stakeholder": True,
            })
    return relations


def extract_support_measures(article: dict) -> List[dict]:
    """(Article)-[:IS_A]->(Support). Support to boost AI."""
    relations = []
    full_text = article.get("full_text", "")
    if not re.search(r"support|promote|foster|subsidize|encourage|designate.*cluster", full_text, re.I):
        return relations
    if re.search(r"Support for|Promotion of|fostering|designation of.*cluster", full_text, re.I):
        relations.append({
            "type": "IS_A",
            "target_node_type": "Support",
            "target_node_name": "AI industry support",
            "description": full_text[:800],
        })
    return relations


def process_item(item: dict, regulation_name: str, all_article_ids: List[str]) -> List[dict]:
    """Extract relations for one article."""
    relations = []
    item_id = item.get("id", "")
    full_text = item.get("full_text", "")

    if "Article 2" in item_id and "Definitions" in full_text:
        relations.extend(extract_article2_defines(item))
        relations.extend(extract_article2_establishes_leads_to(item))
        relations.extend(extract_article2_details_scope(item))

    relations.extend(extract_article_imposes_requirement(item))
    relations.extend(extract_imposes_mandated_for(item))
    relations.extend(extract_permits(item))
    relations.extend(extract_sanctions(item))
    relations.extend(extract_support_measures(item))

    return relations


def main():
    input_path = _PROJECT_ROOT / "data" / "processed" / _SOURCE_FILE
    output_path = _PROJECT_ROOT / "data" / "processed" / _OUTPUT_FILE

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    regulation_name = data.get("source", _REGULATION_NAME)
    articles = data.get("articles") or []
    raw_article_ids = [a["id"] for a in articles if a.get("id")]
    all_article_ids = [_node_id(regulation_name, aid) for aid in raw_article_ids]

    total_relations = 0
    for item in articles:
        raw_id = item.get("id", "")
        item["id"] = _node_id(regulation_name, raw_id)
        item["source_file"] = _SOURCE_FILE_PATH
        item["updated_at"] = _ts()
        rels = process_item(item, regulation_name, all_article_ids)
        item["relations"] = rels
        total_relations += len(rels)

    data["regulation"] = {
        "jurisdiction": "KR",
        "name": regulation_name,
        "effective_date": _EFFECTIVE_DATE,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Extracted {total_relations} relations from {_SOURCE_FILE}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
