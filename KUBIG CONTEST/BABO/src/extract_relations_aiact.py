"""
Extract nodes and relations from AIAct_graph_chunk.json following the Ling Long Protocol
(cursor/rules/relations_extraction.md) for EU AI Act.

Output: AIAct_relations_ready.json with embedded relations per article/rationale/annex.
Schema: Ling Long Protocol. §3.1: (Sanction) MUST connect to (Stakeholder) via [:APPLIES_TO].
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SOURCE_FILE = "AIAct_graph_chunk.json"
_OUTPUT_FILE = "AIAct_relations_ready.json"
_REGULATION_ID = "EU AI Act"
_REGULATION_NAME = "EU AI Act"
_EFFECTIVE_DATE = "2026-08-02"
_SOURCE_FILE_PATH = "data/processed/AIAct_graph_chunk.json"

# Schema: Only legally defined terms as Concepts (Title Form, lang='en')
# Stakeholders from definitions that are persons/bodies — use EXACT same string as Concept
EU_ARTICLE3_DEFINITIONS = [
    ("AI System", "Concept", "en"),
    ("Risk", "Concept", "en"),
    ("Provider", "Concept", "en"),  # Also Stakeholder
    ("Deployer", "Concept", "en"),  # Also Stakeholder
    ("Authorised Representative", "Concept", "en"),  # Also Stakeholder
    ("Importer", "Concept", "en"),  # Also Stakeholder
    ("Distributor", "Concept", "en"),  # Also Stakeholder
    ("Operator", "Concept", "en"),
    ("Product Manufacturer", "Concept", "en"),
    ("Intended Purpose", "Concept", "en"),
    ("Substantial Modification", "Concept", "en"),
    ("General-purpose AI Model", "Concept", "en"),
    ("General-purpose AI System", "Concept", "en"),
    ("High-Risk AI System", "Concept", "en"),
    ("Affected Person", "Concept", "en"),
    ("GPAI Model", "Concept", "en"),
]

# Domains from Annex III (allowed scopes)
DOMAINS = [
    ("Biometrics", "AI systems for biometric identification, categorisation, emotion recognition"),
    ("Critical Infrastructure", "AI systems as safety components in digital infrastructure, traffic, water, gas, heating, electricity"),
    ("Education", "AI systems for access, evaluation, assessment in education"),
    ("Employment", "AI systems for recruitment, work management, self-employment"),
    ("Essential Private/Public Services", "AI systems for eligibility, creditworthiness, insurance, emergency services"),
    ("Healthcare", "AI systems for healthcare and medical services"),
    ("Transportation", "AI systems for road traffic, means of transportation"),
    ("Law Enforcement", "AI systems for law enforcement authorities"),
    ("Migration", "AI systems for migration, asylum, border control"),
    ("Administration of Justice", "AI systems for judicial authority, elections"),
]


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _node_id(regulation: str, raw_id: str) -> str:
    """Article.id format: jurisdiction + law + Article number."""
    return f"{regulation}::{raw_id}"


def _to_title_concept(term: str) -> str:
    """Normalize term to Concept title form (no source tags)."""
    lower = term.lower()
    if "general-purpose ai model" in lower or "general-purpose ai models" in lower:
        return "General-purpose AI Model"
    if "general-purpose ai system" in lower:
        return "General-purpose AI System"
    if "high-risk ai system" in lower or "high-risk ai systems" in lower:
        return "High-Risk AI System"
    if "affected person" in lower or "affected persons" in lower:
        return "Affected Person"
    if "downstream provider" in lower:
        return "Downstream Provider"
    if "product manufacturer" in lower:
        return "Product Manufacturer"
    if "authorised representative" in lower or "authorized representative" in lower:
        return "Authorised Representative"
    # Preserve known abbreviations
    preserve = ("AI", "CE", "EU", "TFEU", "GDPR")
    words = term.split()
    result = []
    for w in words:
        lw = w.lower()
        if lw in ("ai", "ce", "eu") or w.upper() in preserve:
            result.append(w)
        else:
            result.append(w.capitalize() if len(w) > 1 else w)
    return " ".join(result)


def extract_article3_defines(article: dict, regulation_name: str) -> List[dict]:
    """(Article)-[:DEFINES]->(Concept) for Article 3 Definitions."""
    relations = []
    full_text = article.get("full_text", "")
    if "Definitions" not in full_text and "definitions apply" not in full_text.lower():
        return relations
    if "means" not in full_text.lower():
        return relations

    # EU format: (1) 'AI system' means ... or (1) 'term' means
    pattern = r"\(\d+\)\s*['\"]([^'\"]+)['\"]\s+means\s+"
    for m in re.finditer(pattern, full_text):
        term = m.group(1).strip()
        name = _to_title_concept(term)

        # Skip per schema: no daily nouns, no data types, no crimes/examples
        skip_terms = {
            "personal data", "non-personal data", "profiling", "training data",
            "validation data", "testing data", "input data", "biometric data",
            "special categories of personal data", "sensitive operational data",
            "harmonised standard", "common specification", "validation data set",
            "instructions for use", "placing on the market", "making available on the market",
            "putting into service", "safety component", "recall of an ai system",
            "withdrawal of an ai system", "performance of an ai system",
            "conformity assessment", "conformity assessment body", "notified body",
            "notifying authority", "post-market monitoring system", "market surveillance authority",
            "ce marking", "biometric verification", "real-time remote biometric identification system",
            "post-remote biometric identification system", "publicly accessible space",
            "law enforcement authority", "law enforcement", "ai office", "national competent authority",
            "serious incident", "real-world testing plan", "sandbox plan", "ai regulatory sandbox",
            "ai literacy", "testing in real-world conditions", "subject", "informed consent",
            "deep fake", "widespread infringement", "critical infrastructure", "floating-point operation",
        }
        if term.lower() in skip_terms:
            continue

        desc_start = m.end()
        desc_end = full_text.find("\n(", desc_start)
        if desc_end == -1:
            desc_end = min(desc_start + 600, len(full_text))
        excerpt = full_text[desc_start:desc_end].strip()
        if excerpt and len(excerpt) > 20:
            relations.append({
                "type": "DEFINES",
                "target_node_type": "Concept",
                "target_node_name": name,
                "description": excerpt[:800],
            })
    return relations


def extract_rationale_details_definition(rationale: dict) -> List[dict]:
    """(Rationale)-[:DETAILS_DEFINITION_OF]->(Concept)."""
    relations = []
    full_text = rationale.get("full_text", "")
    # Recitals 12-18 detail AI system, deployer, biometric concepts
    concept_keywords = [
        ("AI System", r"notion of ['\"]?AI system['\"]?|AI systems"),
        ("Deployer", r"notion of ['\"]?deployer['\"]?"),
        ("Biometric Data", r"notion of ['\"]?biometric data['\"]?"),
        ("Biometric Identification", r"['\"]?biometric identification['\"]?"),
        ("Biometric Categorisation", r"['\"]?biometric categorisation['\"]?"),
        ("Remote Biometric Identification System", r"['\"]?remote biometric identification system['\"]?"),
        ("Emotion Recognition System", r"['\"]?emotion recognition system['\"]?"),
    ]
    for concept_name, pat in concept_keywords:
        if re.search(pat, full_text, re.I):
            relations.append({
                "type": "DETAILS_DEFINITION_OF",
                "target_node_type": "Concept",
                "target_node_name": concept_name,
                "description": full_text[:1000],
            })
    return relations


def extract_rationale_details_scope(rationale: dict) -> List[dict]:
    """(Rationale)-[:DETAILS_SCOPE]->(Domain)."""
    relations = []
    full_text = rationale.get("full_text", "")
    domain_keywords = [
        ("Healthcare", r"healthcare|health"),
        ("Education", r"education|training"),
        ("Employment", r"employment|workers|labour"),
        ("Law Enforcement", r"law enforcement|biometric identification"),
        ("Critical Infrastructure", r"infrastructure|energy|transport"),
        ("Administration of Justice", r"justice|judicial|democracy"),
        ("Migration", r"migration|asylum|border"),
        ("Biometrics", r"biometric"),
    ]
    for domain_name, pat in domain_keywords:
        if re.search(pat, full_text, re.I):
            relations.append({
                "type": "DETAILS_SCOPE",
                "target_node_type": "Domain",
                "target_node_name": domain_name,
                "description": full_text[:800],
            })
    return relations


def extract_rationale_justifies_logic(rationale: dict, all_article_ids: List[str]) -> List[dict]:
    """(Rationale)-[:JUSTIFIES_LOGIC]->(Article)."""
    relations = []
    full_text = rationale.get("full_text", "")
    if not re.search(r"aims? to|objective|in order to|purpose of", full_text, re.I):
        return relations
    for aid in all_article_ids:
        art_num = re.search(r"Article\s*(\d+)", aid, re.I)
        if art_num and ("Article " + art_num.group(1) in full_text or aid in full_text):
            relations.append({
                "type": "JUSTIFIES_LOGIC",
                "target_node_type": "Article",
                "target_node_name": aid,
                "description": full_text[:800],
            })
    return relations[:3]  # Limit to avoid explosion


def extract_permits(article: dict) -> List[dict]:
    """(Requirement)-[:PERMITS]->(Stakeholder). Trigger: may, allow, be entitled to."""
    relations = []
    full_text = article.get("full_text", "")
    if not re.search(r"\b(may|allow|be entitled to|entitled to)\b", full_text, re.I):
        return relations
    stakeholders = ["Provider", "Deployer", "Affected Person", "Importer", "Distributor"]
    for sh in stakeholders:
        if re.search(rf"\b{sh.lower()}s?\b|\b{sh}\b|affected person", full_text, re.I):
            relations.append({
                "type": "PERMITS",
                "target_node_type": "Stakeholder",
                "target_node_name": sh,
                "description": full_text[:800],
            })
    return relations[:3]


def extract_imposes_mandated_for(article: dict) -> List[dict]:
    """(Article)-[:IMPOSES]->(Requirement)-[:MANDATED_FOR]->(Stakeholder).
    Trigger: shall, must, required to, obliged to.
    """
    relations = []
    full_text = article.get("full_text", "")
    mandatory = re.search(r"\b(shall|must|required to|obliged to)\b", full_text, re.I)
    if not mandatory:
        return relations

    stakeholders = ["Provider", "Deployer", "Importer", "Distributor", "Operator", "Product Manufacturer", "Authorised Representative"]
    for sh in stakeholders:
        if re.search(rf"\b{sh.lower()}s?\b|\b{sh}\b", full_text, re.I):
            req_name = f"Obligation under {article.get('id', '')}"
            relations.append({
                "type": "MANDATED_FOR",
                "target_node_type": "Stakeholder",
                "target_node_name": sh,
                "description": full_text[:800],
            })
    if not relations:
        if "providers" in full_text.lower() or "provider" in full_text.lower():
            relations.append({
                "type": "MANDATED_FOR",
                "target_node_type": "Stakeholder",
                "target_node_name": "Provider",
                "description": full_text[:800],
            })
        if "deployers" in full_text.lower() or "deployer" in full_text.lower():
            relations.append({
                "type": "MANDATED_FOR",
                "target_node_type": "Stakeholder",
                "target_node_name": "Deployer",
                "description": full_text[:800],
            })
    return relations[:5]


def extract_annex_domain_leads_to_risk(annex: dict) -> List[dict]:
    """(Domain)-[:LEADS_TO]->(RiskCategory) from Annex III."""
    relations = []
    full_text = annex.get("full_text", "")
    if "ANNEX III" not in full_text.upper() or "High-risk" not in full_text:
        return relations

    for domain_name, _ in DOMAINS:
        if re.search(re.escape(domain_name), full_text, re.I):
            relations.append({
                "type": "LEADS_TO",
                "start_node_type": "Domain",
                "start_node_name": domain_name,
                "target_node_type": "RiskCategory",
                "target_node_name": "High-Risk AI",
                "description": full_text[:800],
            })
            relations.append({
                "type": "DETAILS_SCOPE",
                "target_node_type": "Domain",
                "target_node_name": domain_name,
                "description": full_text[:800],
            })
    return relations[:10]


def extract_annex_establishes_criterion(annex: dict) -> List[dict]:
    """(Annex)-[:ESTABLISHES]->(UsageCriterion) or TechCriterion."""
    relations = []
    full_text = annex.get("full_text", "")
    aid = annex.get("id", "")

    if "ANNEX III" in aid.upper():
        relations.append({
            "type": "ESTABLISHES",
            "target_node_type": "UsageCriterion",
            "target_node_name": "High-risk AI systems by use area",
            "description": full_text[:800],
        })
    # Annex XIII: TechCriterion for GPAI systemic risk (Article 51)
    if "ANNEX XIII" in aid.upper():
        # (c) FLOPs threshold
        m_flop = re.search(r"amount of computation.*floating point operations", full_text, re.I | re.S)
        if m_flop:
            relations.append({
                "type": "ESTABLISHES",
                "target_node_type": "TechCriterion",
                "target_node_name": "Training computation (FLOPs)",
                "description": m_flop.group(0)[:600],
            })
        # (f) 10 000 business users
        m_users = re.search(r"at least 10\s*000 registered business users", full_text, re.I)
        if m_users:
            relations.append({
                "type": "ESTABLISHES",
                "target_node_type": "TechCriterion",
                "target_node_name": "10 000 registered business users",
                "description": full_text[m_users.start() : m_users.end() + 200],
            })
        # (g) number of registered end-users
        m_end = re.search(r"number of registered end-users", full_text, re.I)
        if m_end:
            relations.append({
                "type": "ESTABLISHES",
                "target_node_type": "TechCriterion",
                "target_node_name": "Number of registered end-users",
                "description": full_text[m_end.start() : m_end.end() + 100],
            })
    return relations


def extract_article_imposes_requirement(article: dict) -> List[dict]:
    """(Article)-[:IMPOSES]->(Requirement)."""
    relations = []
    full_text = article.get("full_text", "")
    if not re.search(r"\b(shall|must|required to|obliged to)\b", full_text, re.I):
        return relations

    req_name = f"Obligation under {article.get('id', '')}"
    relations.append({
        "type": "IMPOSES",
        "target_node_type": "Requirement",
        "target_node_name": req_name,
        "description": full_text[:800],
    })
    return relations


def extract_sanctions(article: dict) -> List[dict]:
    """(Article)-[:PENALIZES_WITH]->(Sanction). §3.1: (Sanction) MUST connect to (Stakeholder) via [:APPLIES_TO]."""
    relations = []
    full_text = article.get("full_text", "")
    # Avoid false positives from definitions (e.g. "criminal penalty"); require amount or sanction keywords
    if not re.search(r"administrative fine|EUR|€|million|imprisonment|fixing a fine|impose a fine", full_text, re.I):
        return relations

    sanction_name = f"Sanction under {article.get('id', '')}"
    relations.append({
        "type": "PENALIZES_WITH",
        "target_node_type": "Sanction",
        "target_node_name": sanction_name,
        "description": full_text[:800],
    })

    # §3.1: (Sanction)-[:APPLIES_TO]->(Stakeholder) — who the sanction applies to
    stakeholders = ["Provider", "Deployer", "Importer", "Distributor", "Authorised Representative", "Operator", "Product Manufacturer"]
    for sh in stakeholders:
        if re.search(rf"\b{re.escape(sh)}\b|{re.escape(sh.lower())}s?\b", full_text, re.I):
            relations.append({
                "type": "APPLIES_TO",
                "start_node_type": "Sanction",
                "start_node_name": sanction_name,
                "target_node_type": "Stakeholder",
                "target_node_name": sh,
                "description": full_text[:800],
            })
    if not any(r.get("type") == "APPLIES_TO" for r in relations):
        # Fallback: if "providers" or "deployers" appear generically
        if re.search(r"\bproviders?\b", full_text, re.I):
            relations.append({
                "type": "APPLIES_TO",
                "start_node_type": "Sanction",
                "start_node_name": sanction_name,
                "target_node_type": "Stakeholder",
                "target_node_name": "Provider",
                "description": full_text[:800],
            })
        elif re.search(r"\bdeployers?\b", full_text, re.I):
            relations.append({
                "type": "APPLIES_TO",
                "start_node_type": "Sanction",
                "start_node_name": sanction_name,
                "target_node_type": "Stakeholder",
                "target_node_name": "Deployer",
                "description": full_text[:800],
            })
        else:
            # §3.1: No Stakeholder extractable from text — use default Provider
            relations.append({
                "type": "APPLIES_TO",
                "start_node_type": "Sanction",
                "start_node_name": sanction_name,
                "target_node_type": "Stakeholder",
                "target_node_name": "Provider",
                "description": full_text[:800] + " [Default Stakeholder: no Stakeholder could be extracted from the text; Provider assigned as default per §3.1.]",
                "is_default_stakeholder": True,
            })
    return relations


def extract_support_measures(article: dict) -> List[dict]:
    """(Article)-[:IS_A]->(Support). Trigger: innovation support, sandbox, SMEs, regulatory sandbox."""
    relations = []
    full_text = article.get("full_text", "")
    item_id = article.get("id", "")
    # Skip Definition articles (Article 3)
    if "Article 3" in item_id and "Definitions" in full_text:
        return relations
    if not re.search(r"measures in support of innovation|regulatory sandbox|AI regulatory sandbox|support.*innovation|innovation.*SME|SME.*innovation", full_text, re.I):
        return relations
    relations.append({
        "type": "IS_A",
        "target_node_type": "Support",
        "target_node_name": "Innovation support",
        "description": full_text[:800],
    })
    return relations


def extract_annex_domain_triggers(annex: dict) -> List[dict]:
    """(Domain)-[:TRIGGERS]->(Requirement). Domain triggers legal obligations (Annex III)."""
    relations = []
    full_text = annex.get("full_text", "")
    if "ANNEX III" not in full_text.upper() or "High-risk" not in full_text:
        return relations
    for domain_name, _ in DOMAINS:
        if re.search(re.escape(domain_name), full_text, re.I):
            relations.append({
                "type": "TRIGGERS",
                "target_node_type": "Requirement",
                "target_node_name": "High-risk AI system requirements",
                "start_node_type": "Domain",
                "start_node_name": domain_name,
                "description": full_text[:600],
            })
    return relations[:10]


def process_item(
    item: dict,
    section: str,
    regulation_name: str,
    all_article_ids: List[str],
) -> List[dict]:
    """Extract relations for one article/rationale/annex."""
    relations = []
    item_id = item.get("id", "")
    full_text = item.get("full_text", "")

    if section == "rationale":
        relations.extend(extract_rationale_details_definition(item))
        relations.extend(extract_rationale_details_scope(item))
        relations.extend(extract_rationale_justifies_logic(item, all_article_ids))

    elif section == "articles":
        if "Definitions" in full_text and "Article 3" in item_id:
            relations.extend(extract_article3_defines(item, regulation_name))
        relations.extend(extract_article_imposes_requirement(item))
        relations.extend(extract_imposes_mandated_for(item))
        relations.extend(extract_permits(item))
        relations.extend(extract_sanctions(item))
        relations.extend(extract_support_measures(item))

    elif section == "annexes":
        relations.extend(extract_annex_domain_leads_to_risk(item))
        relations.extend(extract_annex_establishes_criterion(item))
        relations.extend(extract_annex_domain_triggers(item))
        relations.extend(extract_rationale_details_scope(item))

    return relations


def main():
    input_path = _PROJECT_ROOT / "data" / "processed" / _SOURCE_FILE
    output_path = _PROJECT_ROOT / "data" / "processed" / _OUTPUT_FILE

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    regulation_name = data.get("source", _REGULATION_NAME)
    ts = _ts()

    raw_article_ids = [a["id"] for a in (data.get("articles") or []) if a.get("id")]
    all_article_ids = [_node_id(regulation_name, aid) for aid in raw_article_ids]

    sections = [
        ("rationale", "Rationale"),
        ("articles", "Article"),
        ("annexes", "Annex"),
    ]

    total_relations = 0
    for section_key, _ in sections:
        items = data.get(section_key) or []
        for item in items:
            raw_id = item.get("id", "")
            item["id"] = _node_id(regulation_name, raw_id)
            item["source_file"] = _SOURCE_FILE_PATH
            item["updated_at"] = _ts()
            rels = process_item(item, section_key, regulation_name, all_article_ids)
            item["relations"] = rels
            total_relations += len(rels)

    # Add Regulation node metadata
    data["regulation"] = {
        "jurisdiction": "EU",
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
