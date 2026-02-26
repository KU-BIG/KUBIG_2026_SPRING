"""
EU Legal Ingestion Script for Neo4j AuraDB
Processes the EU AI Act into a parent-child structure optimized for graph database.
Supports: Recitals (Rationale), Articles, and Annexes as parent nodes.
Saves a JSON file for AIAct.html.
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from bs4 import BeautifulSoup


def clean_text(text: str) -> str:
    """
    Clean text by removing excessive whitespace but preserve structure.
    Also removes header/footer patterns.
    """
    # Remove header/footer patterns (e.g., "EN\nOJ L, 12.7.2024\n44/144 ELI: ...")
    text = re.sub(
        r'\nEN\s*\nOJ\s+L[,\s]+\d+\.\d+\.\d+\s*\n\d+/\d+\s+ELI:.*?\n',
        '\n',
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    text = re.sub(
        r'\nEN\s*\nOJ\s+L[,\s]+\d+\.\d+\.\d+.*?\n',
        '\n',
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    text = re.sub(
        r'\n\d+/\d+\s+ELI:.*?\n',
        '\n',
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )

    # Normalize whitespace but keep newlines for structure
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple newlines to double
    return text.strip()


def remove_korean_characters(text: str) -> str:
    """Remove all Korean characters (Hangul) from text using Unicode range [\u3131-\uD79D]."""
    return re.sub(r'[\u3131-\uD79D]', '', text)


def load_pdf_with_page_ranges(
    pdf_path: Path,
    recitals_end: Optional[int] = None,
    articles_end: Optional[int] = None,
) -> Tuple[str, str, str]:
    """
    Load PDF and split into three sections: Recitals, Articles, Annexes.
    """
    reader = PdfReader(pdf_path)
    recitals_text = ""
    articles_text = ""
    annexes_text = ""

    for i, page in enumerate(reader.pages, 1):
        page_text = (page.extract_text() or "") + "\n"

        if recitals_end and i < recitals_end:
            recitals_text += page_text
        elif articles_end and recitals_end and recitals_end <= i < articles_end:
            articles_text += page_text
        elif articles_end and i >= articles_end:
            annexes_text += page_text
        elif recitals_end and i >= recitals_end:
            articles_text += page_text
        else:
            recitals_text += page_text

    return recitals_text, articles_text, annexes_text


def split_recitals(text: str) -> List[Dict[str, Any]]:
    """
    Split recitals text into individual Rationale nodes.
    Uses pattern ^(\\d+) to match recitals at start of line.
    """
    rationales: List[Dict[str, Any]] = []
    recital_pattern = re.compile(r'^\((\d+)\)', re.MULTILINE)

    matches = list(recital_pattern.finditer(text))
    if not matches:
        if text.strip():
            rationales.append({"id": "Recital 1", "full_text": text.strip()})
        return rationales

    for i, match in enumerate(matches):
        recital_num = match.group(1)
        start_pos = match.start()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        recital_text = text[start_pos:end_pos].strip()
        if recital_text:
            rationales.append({"id": f"Recital {recital_num}", "full_text": recital_text})

    return rationales


def assign_chapter_to_article(article_num: int) -> str:
    """
    Assign CHAPTER to Article based on article number.
    Uses the mapping specific to the EU AI Act.
    """
    if 1 <= article_num <= 4:
        return "CHAPTER I"
    elif article_num == 5:
        return "CHAPTER II PROHIBITED AI PRACTICES"
    elif 6 <= article_num <= 49:
        return "CHAPTER III High-Risk AI Systems"
    elif article_num == 50:
        return "CHAPTER IV TRANSPARENCY OBLIGATIONS FOR PROVIDERS AND DEPLOYERS OF CERTAIN AI SYSTEMS"
    elif 51 <= article_num <= 56:
        return "CHAPTER V TRANSPARENCY OBLIGATIONS FOR PROVIDERS AND DEPLOYERS OF CERTAIN AI SYSTEMS"
    elif 57 <= article_num <= 63:
        return "CHAPTER VI MEASURES IN SUPPORT OF INNOVATION"
    elif 64 <= article_num <= 70:
        return "CHAPTER VII GOVERNANCE"
    elif article_num == 71:
        return "CHAPTER VIII EU DATABASE FOR HIGH-RISK AI SYSTEMS"
    elif 72 <= article_num <= 94:
        return "CHAPTER IX POST-MARKET MONITORING, INFORMATION SHARING AND MARKET SURVEILLANCE"
    elif 95 <= article_num <= 96:
        return "CHAPTER X DELEGATION OF POWER AND COMMITTEE PROCEDURE"
    elif 97 <= article_num <= 98:
        return "CHAPTER XI DELEGATION OF POWER AND COMMITTEE PROCEDURE"
    elif 99 <= article_num <= 101:
        return "CHAPTER XII PENALTIES"
    elif 102 <= article_num <= 113:
        return "CHAPTER XIII FINAL PROVISIONS"
    return None


def extract_articles_with_chapter_tracking(text: str) -> List[Dict[str, Any]]:
    """
    Extract Articles with CHAPTER tracking from a raw text version of the EU AI Act.
    Handles PDF spacing issues (e.g., "Ar ticle" -> "Article").
    """
    text = re.sub(r'A\s*r\s*t\s*i\s*c\s*l\s*e', "Article", text, flags=re.IGNORECASE)
    text = re.sub(r'C\s*H\s*A\s*P\s*T\s*E\s*R', "CHAPTER", text, flags=re.IGNORECASE)
    text = re.sub(r'A\s+r\s+t\s+i\s+c\s+l\s+e', "Article", text, flags=re.IGNORECASE)
    text = re.sub(r'C\s+h\s+a\s+p\s+t\s+e\s+r', "CHAPTER", text, flags=re.IGNORECASE)

    lines = text.split("\n")
    articles: List[Dict[str, Any]] = []
    current_chapter: Optional[str] = None
    current_article_id: Optional[str] = None
    current_article_lines: List[str] = []

    chapter_pattern = re.compile(r"^C\s*H\s*A\s*P\s*T\s*E\s*R\s+(\d+)", re.IGNORECASE)
    article_pattern = re.compile(r"^A\s*r\s*t\s*i\s*c\s*l\s*e\s+(\d+)", re.IGNORECASE)

    def save_current_article() -> None:
        nonlocal current_article_id, current_article_lines, articles
        if not current_article_id or not current_article_lines:
            return
        article_text = "\n".join(current_article_lines).strip()
        article_text = re.sub(
            r"A\s+r\s+t\s+i\s+c\s+l\s+e", "Article", article_text, flags=re.IGNORECASE
        )
        article_num = int(current_article_id)
        assigned_chapter = assign_chapter_to_article(article_num)
        final_chapter = current_chapter if current_chapter else assigned_chapter
        articles.append(
            {
                "id": f"Article {current_article_id}",
                "metadata": {"chapter": final_chapter},
                "full_text": article_text,
            }
        )
        current_article_lines = []

    for line in lines:
        line_stripped = line.strip()
        chapter_match = chapter_pattern.match(line_stripped)
        if chapter_match:
            save_current_article()
            current_chapter = f"CHAPTER {chapter_match.group(1)}"
            continue

        article_match = article_pattern.match(line_stripped)
        if article_match:
            save_current_article()
            current_article_id = article_match.group(1)
            normalized_line = re.sub(
                r"A\s+r\s+t\s+i\s+c\s+l\s+e", "Article", line_stripped, flags=re.IGNORECASE
            )
            current_article_lines = [normalized_line]
            continue

        if current_article_id:
            normalized_line = re.sub(
                r"A\s+r\s*t\s*i\s*c\s*l\s+e", "Article", line, flags=re.IGNORECASE
            )
            current_article_lines.append(normalized_line)

    save_current_article()
    return articles


def extract_annexes(text: str) -> List[Dict[str, Any]]:
    """
    Extract Annexes from plain text.
    Looks for headers like "ANNEX I", "ANNEX II", etc. (Roman numerals).
    """
    annexes: List[Dict[str, Any]] = []
    annex_pattern = re.compile(r"^ANNEX\s+([IVX]+)", re.IGNORECASE | re.MULTILINE)

    matches = list(annex_pattern.finditer(text))
    if not matches:
        return annexes

    for i, match in enumerate(matches):
        annex_num = match.group(1)
        start_pos = match.start()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        annex_text = text[start_pos:end_pos].strip()

        title = None
        for line in annex_text.split("\n")[:10]:
            line_clean = line.strip()
            if line_clean and not re.match(
                r"^ANNEX\s+[IVX]+", line_clean, re.IGNORECASE
            ):
                if 10 < len(line_clean) < 200:
                    title = line_clean
                    break

        annexes.append(
            {
                "id": f"ANNEX {annex_num}",
                "title": title or f"Annex {annex_num}",
                "full_text": annex_text,
            }
        )

    return annexes


def create_chunks_with_structure(
    parent_text: str,
    parent_id: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    custom_separators: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """
    Create vector chunks from parent text, respecting paragraph structure.
    """
    if not parent_text or not parent_text.strip():
        return []

    clean_id = parent_id.replace(" ", "_").replace("ANNEX", "annex").lower()

    if len(parent_text) <= chunk_size:
        return [{"chunk_id": f"{clean_id}_c1", "text": parent_text.strip()}]

    separators = (
        custom_separators if custom_separators else ["\n\n", "\n", ". ", " ", ""]
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=separators,
    )

    chunks: List[Dict[str, str]] = []
    for i, chunk_text in enumerate(text_splitter.split_text(parent_text), 1):
        chunks.append({"chunk_id": f"{clean_id}_c{i}", "text": chunk_text.strip()})

    return chunks


def process_aiact_html(html_path: Path) -> Dict[str, Any]:
    """
    Process AIAct.html using the official OJ HTML structure.
    Produces rationale (recitals), articles, and annexes with chunks.
    """
    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    adoption_elem = None
    for p in soup.find_all("p", class_="oj-normal"):
        if p.get_text() and "HAVE ADOPTED THIS REGULATION" in p.get_text():
            adoption_elem = p
            break

    all_elements = soup.find_all("p")

    found_adoption = False
    current_chapter = None
    current_article_id = None
    current_article_text: List[str] = []
    current_recital_num = None
    current_recital_text: List[str] = []
    current_annex_num = None
    current_annex_text: List[str] = []

    rationales: List[Dict[str, Any]] = []
    articles: List[Dict[str, Any]] = []
    annexes: List[Dict[str, Any]] = []

    def get_element_text(elem) -> str:
        if not elem:
            return ""
        return elem.get_text(separator="\n", strip=True).strip()

    for elem in all_elements:
        text = get_element_text(elem)
        if not text:
            continue

        if adoption_elem and elem == adoption_elem:
            found_adoption = True
            if current_recital_num and current_recital_text:
                rationales.append(
                    {
                        "id": f"Recital {current_recital_num}",
                        "full_text": "\n".join(current_recital_text).strip(),
                    }
                )
                current_recital_text = []
            continue

        if not found_adoption:
            recital_match = re.match(r"^\((\d+)\)", text)
            if recital_match:
                if current_recital_num and current_recital_text:
                    rationales.append(
                        {
                            "id": f"Recital {current_recital_num}",
                            "full_text": "\n".join(current_recital_text).strip(),
                        }
                    )
                current_recital_num = recital_match.group(1)
                current_recital_text = [text]
            elif current_recital_num and text:
                current_recital_text.append(text)
            continue

        # After adoption: process articles / annexes
        if current_annex_num:
            if (
                "oj-doc-ti" in elem.get("class", [])
                and re.match(r"^ANNEX\s+([IVX]+)", text, re.I)
            ) or (
                "oj-ti-art" in elem.get("class", [])
                and re.match(r"^Article\s+\d+", text, re.I)
            ):
                if current_annex_text:
                    annexes.append(
                        {
                            "id": f"ANNEX {current_annex_num}",
                            "title": f"Annex {current_annex_num}",
                            "full_text": "\n".join(current_annex_text).strip(),
                        }
                    )
                    current_annex_text = []
                    current_annex_num = None
            else:
                current_annex_text.append(text)
                continue

        if "oj-doc-ti" in elem.get("class", []):
            annex_match = re.match(r"^ANNEX\s+([IVX]+)", text, re.I)
            if annex_match:
                if current_article_id and current_article_text:
                    article_num = int(current_article_id)
                    assigned_chapter = assign_chapter_to_article(article_num)
                    final_chapter = current_chapter or assigned_chapter
                    articles.append(
                        {
                            "id": f"Article {current_article_id}",
                            "metadata": {"chapter": final_chapter},
                            "full_text": "\n".join(current_article_text).strip(),
                        }
                    )
                    current_article_text = []
                    current_article_id = None

                current_annex_num = annex_match.group(1)
                current_annex_text = [text]
                continue

        if "oj-ti-section-2" in elem.get("class", []):
            continue

        if "oj-ti-section-1" in elem.get("class", []):
            chapter_match = re.match(r"^CHAPTER\s+([IVX]+|\d+)", text, re.I)
            if chapter_match:
                if current_article_id and current_article_text:
                    article_num = int(current_article_id)
                    assigned_chapter = assign_chapter_to_article(article_num)
                    final_chapter = current_chapter or assigned_chapter
                    articles.append(
                        {
                            "id": f"Article {current_article_id}",
                            "metadata": {"chapter": final_chapter},
                            "full_text": "\n".join(current_article_text).strip(),
                        }
                    )
                    current_article_text = []
                    current_article_id = None

                chapter_val = chapter_match.group(1)
                if chapter_val.isdigit():
                    current_chapter = f"CHAPTER {chapter_val}"
                else:
                    current_chapter = text.strip()
                continue

        if "oj-ti-art" in elem.get("class", []):
            article_match = re.match(r"^Article\s+(\d+)", text, re.I)
            if article_match:
                if current_article_id and current_article_text:
                    article_num = int(current_article_id)
                    assigned_chapter = assign_chapter_to_article(article_num)
                    final_chapter = current_chapter or assigned_chapter
                    articles.append(
                        {
                            "id": f"Article {current_article_id}",
                            "metadata": {"chapter": final_chapter},
                            "full_text": "\n".join(current_article_text).strip(),
                        }
                    )

                current_article_id = article_match.group(1)
                current_article_text = [text]
                continue

        if (
            current_article_id
            and text
            and "oj-ti-section" not in " ".join(elem.get("class", []))
        ):
            current_article_text.append(text)

    if current_recital_num and current_recital_text:
        rationales.append(
            {
                "id": f"Recital {current_recital_num}",
                "full_text": "\n".join(current_recital_text).strip(),
            }
        )

    if current_article_id and current_article_text:
        article_num = int(current_article_id)
        assigned_chapter = assign_chapter_to_article(article_num)
        final_chapter = current_chapter or assigned_chapter
        articles.append(
            {
                "id": f"Article {current_article_id}",
                "metadata": {"chapter": final_chapter},
                "full_text": "\n".join(current_article_text).strip(),
            }
        )

    if current_annex_num and current_annex_text:
        annexes.append(
            {
                "id": f"ANNEX {current_annex_num}",
                "title": f"Annex {current_annex_num}",
                "full_text": "\n".join(current_annex_text).strip(),
            }
        )

    for rationale in rationales:
        rationale["full_text"] = clean_text(rationale["full_text"])
        rationale["chunks"] = create_chunks_with_structure(
            rationale["full_text"], rationale["id"]
        )

    article_separators = ["\n\n", "\nArticle ", "\n", ". ", " "]
    for article in articles:
        article["full_text"] = clean_text(article["full_text"])
        article["chunks"] = create_chunks_with_structure(
            article["full_text"], article["id"], custom_separators=article_separators
        )

    for annex in annexes:
        annex["full_text"] = clean_text(annex["full_text"])
        annex["chunks"] = create_chunks_with_structure(annex["full_text"], annex["id"])

    return {
        "source": "EU AI Act",
        "rationale": rationales,
        "articles": articles,
        "annexes": annexes,
    }


def main() -> None:
    """
    Main entrypoint: process AIAct.html and emit AIAct_graph_chunk.json.
    """
    import json

    project_root = Path(__file__).parent.parent
    raw_data_dir = project_root / "data" / "raw"
    output_dir = project_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not raw_data_dir.exists():
        print(f"Error: Directory {raw_data_dir} does not exist!")
        return

    aiact_path = raw_data_dir / "AIAct.html"
    if aiact_path.exists():
        print(f"\nProcessing: {aiact_path.name}")
        try:
            aiact_data = process_aiact_html(aiact_path)

            num_rationale = len(aiact_data["rationale"])
            num_articles = len(aiact_data["articles"])
            num_annexes = len(aiact_data["annexes"])
            total_chunks = (
                sum(len(r["chunks"]) for r in aiact_data["rationale"])
                + sum(len(a["chunks"]) for a in aiact_data["articles"])
                + sum(len(an["chunks"]) for an in aiact_data["annexes"])
            )

            print(
                f"Found {num_rationale} Rationale, {num_articles} Articles, "
                f"{num_annexes} Annexes, and {total_chunks} Chunks."
            )

            output_file = output_dir / "AIAct_graph_chunk.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(aiact_data, f, ensure_ascii=False, indent=2)
            print(f"✓ Saved to: {output_file}")

            if aiact_data["annexes"]:
                sample_annex = aiact_data["annexes"][0]
                print(f"\n{'=' * 60}")
                print("EXAMPLE ANNEX NODE:")
                print(f"{'=' * 60}")
                print(
                    json.dumps(
                        {
                            "id": sample_annex["id"],
                            "title": sample_annex.get("title", "N/A"),
                            "full_text_preview": (
                                sample_annex["full_text"][:200] + "..."
                                if len(sample_annex["full_text"]) > 200
                                else sample_annex["full_text"]
                            ),
                            "chunks_count": len(sample_annex["chunks"]),
                            "first_chunk_id": (
                                sample_annex["chunks"][0]["chunk_id"]
                                if sample_annex["chunks"]
                                else None
                            ),
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                )

        except Exception as e:
            print(f"Error processing {aiact_path.name}: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()

