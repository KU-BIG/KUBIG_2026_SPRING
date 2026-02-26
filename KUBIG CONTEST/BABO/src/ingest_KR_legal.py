"""
Standalone ingestion script for AICoreLawEN.pdf (Korea AI Law, English-only PDF).
Fixes: false-positive Article headers, CHAPTER/SECTION header leakage,
item-unaware chunking, and residual artifacts.
Output: AICoreLaw_graph_chunk.json with articles and item-aware chunks.
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


# ---- Pre-processing (cleaning) ----

def remove_circled_numbers(text: str) -> str:
    """Remove circled numbers ①-⑳ (\\u2460-\\u2473)."""
    return re.sub(r'[\u2460-\u2473]', '', text)


def remove_orphan_punctuation_lines(text: str) -> str:
    """Remove lines containing only . , ) ( or whitespace."""
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append(line)
            continue
        if re.match(r'^[\s.,)(]+$', stripped):
            continue
        cleaned.append(line)
    return '\n'.join(cleaned)


def merge_split_article_headers(text: str) -> str:
    """Merge split article headers into one line."""
    # "Article N" on one line, "(Title)" on next -> "Article N (Title)"
    text = re.sub(
        r'(Article\s+\d+)\s*\n\s*\(([^)]+)\)',
        r'\1 (\2)',
        text,
    )
    # "Article" on one line, "N (Title)" on next (e.g. PDF line break after "Article") -> "Article N (Title)"
    text = re.sub(
        r'(Article)\s*\n\s*(\d+)\s*\(([^)]+)\)',
        r'\1 \2 (\3)',
        text,
    )
    # Title split across lines: "Article N (First part" then "rest)" -> "Article N (First part rest)"
    lines = text.split('\n')
    merged: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Line starts with "Article N (" but has no closing ")" on this line
        m = re.match(r'^(Article\s+\d+\s*\([^)]*)$', line.strip(), re.IGNORECASE)
        if m and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            # Next line completes the paren (e.g. "intelligence)" or "rest of title)")
            if ')' in next_line:
                # Merge: "Article 30 (artificial" + "intelligence)" -> "Article 30 (artificial intelligence)"
                closing = next_line.index(')')
                suffix = next_line[: closing + 1]
                rest = next_line[closing + 1 :].strip()
                merged.append(m.group(1) + ' ' + suffix)
                if rest:
                    merged.append(rest)
                i += 2
                continue
        merged.append(line)
        i += 1
    return '\n'.join(merged)


def preprocess(raw_text: str) -> str:
    """Apply all pre-processing steps."""
    # AICoreLawEN.pdf is English-only, so we skip Korean-character removal.
    text = raw_text
    text = remove_circled_numbers(text)
    text = remove_orphan_punctuation_lines(text)
    text = merge_split_article_headers(text)
    # Normalize spacing (e.g. PDF "Ar ticle" -> "Article")
    text = re.sub(r'A\s*r\s*t\s*i\s*c\s*l\s*e', 'Article', text, flags=re.IGNORECASE)
    text = re.sub(r'C\s*H\s*A\s*P\s*T\s*E\s*R', 'CHAPTER', text, flags=re.IGNORECASE)
    text = re.sub(r'S\s*E\s*C\s*T\s*I\s*O\s*N', 'SECTION', text, flags=re.IGNORECASE)
    text = re.sub(r'A\s*D\s*D\s*E\s*N\s*D\s*A', 'ADDENDA', text, flags=re.IGNORECASE)
    return text


# ---- State machine (parsing) ----

def parse_articles_with_state_machine(text: str) -> List[Dict[str, Any]]:
    """
    Iterate line-by-line. Track Chapter, Section, and Addenda; only add body lines to article buffer.
    ADDENDA is a distinct chapter; section is reset when CHAPTER or ADDENDA starts.
    Article IDs in Addenda are "Addenda Article N" to avoid conflict with main "Article N".
    """
    lines = text.split('\n')
    articles: List[Dict[str, Any]] = []

    current_chapter: Optional[str] = None
    current_section: Optional[str] = None
    is_addenda: bool = False
    current_article_id: Optional[str] = None
    current_article_title: Optional[str] = None
    article_buffer: List[str] = []
    in_chapter_header = False
    in_section_header = False

    addenda_re = re.compile(r'^ADDENDA\b', re.IGNORECASE)  # e.g. "ADDENDA" or "ADDENDA <Act No. ...>"
    chapter_re = re.compile(r'^CHAPTER\s+[IVX]+', re.IGNORECASE)
    section_re = re.compile(r'^SECTION\s+\d+', re.IGNORECASE)
    # Strict Article header: standalone line, title in parens with at least one letter
    article_re = re.compile(
        r'^Article\s+(\d+)\s*\(([^)]*[A-Za-z][^)]*)\)\s*$',
        re.IGNORECASE,
    )
    # Fallback for Article 30 when PDF has different formatting
    article_30_fallback_re = re.compile(
        r'^Article\s+(30)\s*\(([^)]*)\)',  # "Article 30 (Title)" with optional trailing or empty title
        re.IGNORECASE,
    )
    article_30_bare_re = re.compile(
        r'^Article\s+(30)\s*$',  # line is only "Article 30" (no parens)
        re.IGNORECASE,
    )

    def flush_article() -> None:
        nonlocal article_buffer, current_article_id, current_article_title, articles
        if not current_article_id or not article_buffer:
            return
        full_text = '\n'.join(article_buffer).strip()
        if not full_text:
            return
        metadata: Dict[str, Any] = {}
        if current_chapter is not None:
            metadata['chapter'] = current_chapter
        metadata['section'] = current_section  # Explicit None when no section
        if current_article_title is not None:
            metadata['title'] = current_article_title.strip()
        # ID: "Addenda Article N" in Addenda to avoid conflict with main "Article N"
        article_id = f'Addenda Article {current_article_id}' if is_addenda else f'Article {current_article_id}'
        articles.append({
            'id': article_id,
            'metadata': metadata,
            'full_text': full_text,
        })
        article_buffer = []
        current_article_title = None

    for line in lines:
        line_stripped = line.strip()

        # Rule A: ADDENDA (check before CHAPTER)
        if addenda_re.match(line_stripped):
            flush_article()
            current_chapter = 'ADDENDA'
            current_section = None
            is_addenda = True
            current_article_id = None  # So lines until next Article are not appended
            in_chapter_header = False
            in_section_header = False
            continue

        # Rule B: CHAPTER
        if chapter_re.match(line_stripped):
            flush_article()
            current_chapter = line_stripped
            current_section = None  # Force reset to prevent section leakage
            is_addenda = False
            current_article_id = None
            in_chapter_header = True
            in_section_header = False
            continue

        # Rule C: SECTION
        if section_re.match(line_stripped):
            flush_article()
            current_section = line_stripped
            in_section_header = True
            in_chapter_header = False
            continue

        # Rule D: ARTICLE header (must have title in parens)
        art_match = article_re.match(line_stripped)
        if not art_match:
            art_match = article_30_fallback_re.match(line_stripped)  # "Article 30 (Title)" with odd format
        if not art_match:
            bare_30 = article_30_bare_re.match(line_stripped)  # line is only "Article 30"
            if bare_30:
                art_match = bare_30  # use group(1)=30, no group(2)
        if art_match:
            flush_article()
            in_chapter_header = False
            in_section_header = False
            current_article_id = art_match.group(1)
            current_article_title = (art_match.group(2).strip() or None) if art_match.lastindex >= 2 else None
            article_buffer = [line_stripped]
            continue

        # Continuation of CHAPTER title (do not add to article buffer)
        if in_chapter_header:
            if line_stripped:
                current_chapter = (current_chapter or '') + '\n' + line_stripped
            continue

        # Continuation of SECTION title (do not add to article buffer)
        if in_section_header:
            if line_stripped:
                current_section = (current_section or '') + '\n' + line_stripped
            continue

        # Rule E: Text content
        if current_article_id is not None:
            article_buffer.append(line)

    flush_article()
    return articles


# ---- Item-aware chunking ----

ITEM_PATTERN = re.compile(r'(?m)^\s*(\d+)\.\s*')
ITEM_SPLIT_PATTERN = re.compile(r'\n(?=\d+\.\s+)')


def process_article_chunks(full_text: str, article_id: str) -> List[Dict[str, str]]:
    """
    If article has numbered list (1., 2., 3.), split by item; else use RecursiveCharacterTextSplitter(1000).
    Chunk 0 = preamble (text before first '1.'), Chunk 1 = '1. ' + item1, etc.
    """
    if not full_text or not full_text.strip():
        return []

    # Normalize article_id for chunk_id: "Article 13" -> art13, "Addenda Article 1" -> addenda_art1
    if article_id.startswith('Addenda Article '):
        num = article_id.replace('Addenda Article ', '').strip()
        base = f'addenda_art{num}'
    else:
        art_num = article_id.replace('Article ', '').strip()
        base = f'art{art_num}'

    if not ITEM_PATTERN.search(full_text):
        # No list: use RecursiveCharacterTextSplitter size=1000
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        sub_chunks = splitter.split_text(full_text)
        return [
            {'chunk_id': f'{base}_c{i}', 'text': t.strip()}
            for i, t in enumerate(sub_chunks, 1)
        ]

    # Split by item: preamble + "1. ...", "2. ...", ...
    parts = ITEM_SPLIT_PATTERN.split(full_text)
    # First part is preamble (may include "Article N (Title)" and text before "1.")
    preamble = parts[0].strip()
    chunks: List[Dict[str, str]] = []

    if preamble:
        chunks.append({'chunk_id': f'{base}_item0', 'text': preamble})

    for i, block in enumerate(parts[1:], 1):
        block = block.strip()
        if not block:
            continue
        chunks.append({'chunk_id': f'{base}_item{i}', 'text': block})

    return chunks


# ---- PDF load and pipeline ----

def load_pdf_text(pdf_path: Path, skip_addenda: bool = False) -> str:
    """Load full text from PDF. Include ADDENDA pages so they can be parsed as a distinct chapter."""
    reader = PdfReader(pdf_path)
    full = []
    for i, page in enumerate(reader.pages):
        if skip_addenda and i in (38, 39):
            continue
        full.append(page.extract_text() or '')
    return '\n'.join(full)


def process_aicorelaw(pdf_path: Path) -> Dict[str, Any]:
    """
    Full pipeline: load PDF -> preprocess -> state-machine parse -> item-aware chunks.
    Returns dict with source and articles only (no rationale/annexes).
    """
    raw = load_pdf_text(pdf_path)
    text = preprocess(raw)
    articles = parse_articles_with_state_machine(text)

    for art in articles:
        art['chunks'] = process_article_chunks(art['full_text'], art['id'])

    return {
        'source': 'Korea AI Law',
        'articles': articles,
    }


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / 'data' / 'raw'
    out_dir = project_root / 'data' / 'processed'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prefer the English-only PDF; fall back to the original name if needed.
    pdf_path = raw_dir / 'AICoreLawEN.pdf'
    if not pdf_path.exists():
        fallback = raw_dir / 'AICoreLaw.pdf'
        if fallback.exists():
            pdf_path = fallback
        else:
            print(f"Error: {pdf_path} not found (also missing: {fallback.name}).")
            return

    print(f"Processing: {pdf_path.name}")
    data = process_aicorelaw(pdf_path)
    num_articles = len(data['articles'])
    total_chunks = sum(len(a['chunks']) for a in data['articles'])
    print(f"Articles: {num_articles}, Chunks: {total_chunks}")

    out_file = out_dir / 'AICoreLaw_graph_chunk.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved: {out_file}")


if __name__ == '__main__':
    main()
