"""
Prepare fine-tuning corpus from relations JSON and guideline PDFs.

Output: data/finetune/corpus.jsonl in unified format (see data/finetune/FORMAT_SPEC.md).

Usage:
  python src/prepare_finetune_data.py [--relations-only] [--guidelines-only]
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_AIACT_PATH = _PROJECT_ROOT / "data" / "processed" / "AIAct_relations_ready.json"
_AICORELAW_PATH = _PROJECT_ROOT / "data" / "processed" / "AICoreLaw_relations_ready.json"
_GUIDELINES_DIR = _PROJECT_ROOT / "data" / "guidelines"
_OUTPUT_PATH = _PROJECT_ROOT / "data" / "finetune" / "corpus.jsonl"

# Domains per relations_extraction.md (for guideline PDF scanning)
DOMAIN_KEYWORDS_EN = [
    "Biometrics",
    "Critical Infrastructure",
    "Education",
    "Employment",
    "Essential Private Services",
    "Essential Public Services",
    "Healthcare",
    "Transportation",
    "Law Enforcement",
    "Migration",
    "Administration of Justice",
]
DOMAIN_KEYWORDS_KR = [
    "생체정보",
    "핵심기반시설",
    "교육",
    "고용",
    "필수민간서비스",
    "필수공공서비스",
    "헬스케어",
    "의료",
    "교통",
    "수송",
    "법집행",
    "이민",
    "국경",
    "사법",
    "금융",
    "신용",
]


def _normalize_article_ref(item_id: str, section_type: str) -> str:
    """Extract article reference from item id (e.g. EU AI Act::Article 71 -> Article 71)."""
    if not item_id:
        return ""
    # Strip regulation prefix
    for prefix in ("EU AI Act::", "Korea AI Law::", "AIAct::", "KRAILaw::"):
        if item_id.startswith(prefix):
            return item_id[len(prefix) :].strip()
    return item_id


def _section_type_from_item(item: Dict, section_key: str, item_id: str) -> str:
    """Map section key and item id to section_type."""
    if section_key == "rationale":
        return "recitals"
    if section_key == "annexes":
        return "annex"
    if section_key == "articles":
        if item_id and "Addenda Article" in item_id:
            return "addenda"
        return "article"
    return "article"


def process_relations_item(
    item: Dict,
    section_key: str,
    regulation: str,
) -> List[Dict[str, Any]]:
    """
    Extract Sanction and Requirement relation descriptions from one item.
    Returns list of corpus entries in unified format. Merges all descs per item
    for adequate context length.
    """
    entries: List[Dict[str, Any]] = []
    item_id = item.get("id", "")
    section_type = _section_type_from_item(item, section_key, item_id)
    article_ref = _normalize_article_ref(item_id, section_type)

    rels = item.get("relations") or []
    sanction_descs: List[str] = []
    requirement_descs: List[str] = []

    for rel in rels:
        target_type = rel.get("target_node_type") or ""
        start_type = rel.get("start_node_type") or ""
        desc = (rel.get("description") or "").strip()
        if not desc or len(desc) < 20:
            continue
        if target_type == "Sanction" or start_type == "Sanction":
            if desc not in sanction_descs:
                sanction_descs.append(desc)
        if target_type == "Requirement":
            if desc not in requirement_descs:
                requirement_descs.append(desc)

    if not sanction_descs and not requirement_descs:
        return []

    node_types: List[str] = []
    if sanction_descs:
        node_types.append("Sanction")
    if requirement_descs:
        node_types.append("Requirement")

    merged_parts = sanction_descs + requirement_descs
    header = f"[{regulation}] [{section_type.upper()}] {article_ref}\n\n"
    merged_text = header + "\n\n".join(merged_parts)

    if len(merged_text) < 80:
        return []

    source_file = "AIAct_relations_ready.json" if regulation == "AIAct" else "AICoreLaw_relations_ready.json"
    meta: Dict[str, Any] = {
        "regulation": regulation,
        "section_type": section_type,
        "article_ref": article_ref,
        "source_file": source_file,
        "node_types": node_types,
    }
    entries.append({"meta": meta, "text": merged_text})
    return entries


def process_relations_json(regulation: str, data: Dict) -> Iterator[Dict[str, Any]]:
    """Yield corpus entries from relations JSON for one regulation."""
    sections = [
        ("rationale", "recitals"),
        ("articles", "article"),
        ("annexes", "annex"),
    ]
    if regulation == "KRAILaw":
        sections = [("articles", "article")]  # KR has only articles (incl. addenda)

    for section_key, _ in sections:
        items = data.get(section_key) or []
        for item in items:
            for entry in process_relations_item(item, section_key, regulation):
                yield entry


def run_relations(output_path: Path) -> int:
    """Process AIAct and AICoreLaw relations. Returns count of entries written."""
    count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # AI Act
    if _AIACT_PATH.exists():
        with open(_AIACT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        with open(output_path, "a", encoding="utf-8") as out:
            for entry in process_relations_json("AIAct", data):
                out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1
    else:
        print(f"  [skip] {_AIACT_PATH} not found")

    # Korea AI Law
    if _AICORELAW_PATH.exists():
        with open(_AICORELAW_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        with open(output_path, "a", encoding="utf-8") as out:
            for entry in process_relations_json("KRAILaw", data):
                out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1
    else:
        print(f"  [skip] {_AICORELAW_PATH} not found")

    return count


def extract_domain_sections_from_text(text: str, lang: str = "en") -> List[Tuple[str, str, Optional[str]]]:
    """
    Find sections that mention Domain keywords. Returns list of (section_text, domain, article_ref).
    """
    results: List[Tuple[str, str, Optional[str]]] = []
    keywords = DOMAIN_KEYWORDS_KR if lang == "ko" else DOMAIN_KEYWORDS_EN
    # Also try to detect article references
    article_pattern = re.compile(
        r"(?:Article\s+\d+|제\s*\d+\s*조|Art\.\s*\d+)",
        re.IGNORECASE,
    )

    # Simple paragraph splitting (double newline or large block)
    paras = re.split(r"\n\s*\n", text)
    for para in paras:
        para = para.strip()
        if len(para) < 100:
            continue
        found_domains = [kw for kw in keywords if kw in para or kw.lower() in para.lower()]
        article_match = article_pattern.search(para)
        article_ref = article_match.group(0) if article_match else None
        for domain in found_domains:
            results.append((para, domain, article_ref))
    return results


def infer_guideline_regulation(filename: str, text_sample: str) -> str:
    """Infer regulation tag from filename and content."""
    name_lower = filename.lower()
    if "aiact" in name_lower or "eu" in name_lower or "european" in name_lower or "commission" in name_lower:
        return "AIAct"
    if "krai" in name_lower or "korea" in name_lower or "pipc" in name_lower or "fsc" in name_lower or "금융" in name_lower or "개인정보" in name_lower:
        return "KRAILaw"
    if any(c >= "\uAC00" for c in text_sample[:500]):
        return "KRAILaw"
    return "AIAct"


def infer_guideline_source(filename: str) -> str:
    """Infer guideline source (PIPC, FSC, EU Commission, BSA)."""
    name = filename.lower()
    if "pipc" in name or "개인정보" in name:
        return "PIPC"
    if "fsc" in name or "금융" in name:
        return "FSC"
    if "commission" in name or "eu" in name:
        return "EU Commission"
    if "bsa" in name:
        return "BSA"
    return "Unknown"


def process_pdf_guidelines(pdf_path: Path) -> List[Dict[str, Any]]:
    """Extract Domain-relevant sections from a guideline PDF."""
    try:
        from pypdf import PdfReader
    except ImportError:
        print("  [skip] pypdf not installed; pip install pypdf")
        return []

    entries: List[Dict[str, Any]] = []
    reader = PdfReader(str(pdf_path))
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() or ""
        full_text += "\n"

    if len(full_text.strip()) < 200:
        return []

    lang = "ko" if any("\uAC00" <= c <= "\uD7A3" for c in full_text[:1000]) else "en"
    regulation = infer_guideline_regulation(pdf_path.name, full_text)
    guideline_source = infer_guideline_source(pdf_path.name)

    sections = extract_domain_sections_from_text(full_text, lang)
    seen = set()
    for para, domain, article_ref in sections:
        key = (para[:200], domain)
        if key in seen:
            continue
        seen.add(key)
        header = f"[{regulation}] [GUIDELINE] {guideline_source} [Domain: {domain}]"
        if article_ref:
            header += f" [{article_ref}]"
        header += "\n\n"
        text = header + para
        if len(text) < 100:
            continue
        meta = {
            "regulation": regulation,
            "section_type": "guideline",
            "article_ref": article_ref or "",
            "source_file": pdf_path.name,
            "domain": domain,
            "guideline_source": guideline_source,
        }
        entries.append({"meta": meta, "text": text})

    # If no Domain sections found, check for article-specific content and use larger chunks
    if not entries:
        article_pattern = re.compile(
            r"(?:Article\s+\d+|제\s*\d+\s*조)",
            re.IGNORECASE,
        )
        if article_pattern.search(full_text):
            for match in article_pattern.finditer(full_text):
                start = max(0, match.start() - 200)
                end = min(len(full_text), match.end() + 800)
                chunk = full_text[start:end].strip()
                if len(chunk) >= 150:
                    header = f"[{regulation}] [GUIDELINE] {guideline_source} [Article-specific]\n\n"
                    text = header + chunk
                    meta = {
                        "regulation": regulation,
                        "section_type": "guideline",
                        "article_ref": match.group(0),
                        "source_file": pdf_path.name,
                        "guideline_source": guideline_source,
                    }
                    entries.append({"meta": meta, "text": text})
                    break  # One article-specific entry per doc

    return entries


def run_guidelines(output_path: Path) -> int:
    """Process guideline PDFs. Returns count of entries written."""
    if not _GUIDELINES_DIR.exists():
        print(f"  [skip] {_GUIDELINES_DIR} does not exist")
        return 0

    count = 0
    pdfs = list(_GUIDELINES_DIR.glob("**/*.pdf"))
    if not pdfs:
        print(f"  [skip] No PDFs in {_GUIDELINES_DIR}")
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as out:
        for pdf_path in pdfs:
            for entry in process_pdf_guidelines(pdf_path):
                out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare fine-tuning corpus")
    parser.add_argument("--relations-only", action="store_true", help="Process only relations JSON")
    parser.add_argument("--guidelines-only", action="store_true", help="Process only guideline PDFs")
    parser.add_argument("-o", "--output", default=str(_OUTPUT_PATH), help="Output JSONL path")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.write_text("")  # Clear for fresh run

    total = 0
    if not args.guidelines_only:
        print("Processing relations JSON...")
        n = run_relations(output_path)
        total += n
        print(f"  Relations: {n} entries")

    if not args.relations_only:
        print("Processing guideline PDFs...")
        n = run_guidelines(output_path)
        total += n
        print(f"  Guidelines: {n} entries")

    print(f"\nTotal: {total} entries -> {output_path}")


if __name__ == "__main__":
    main()
