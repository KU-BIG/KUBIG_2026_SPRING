# Unified Fine-Tuning Data Format

This document specifies the format for all fine-tuning corpus entries, used for continued pre-training (causal LM) on AI compliance and related regulatory text.

## Output Format: JSONL

Each line is a JSON object. One document per line.

```json
{
  "meta": {
    "section_type": "article",
    "article_ref": "Article 71",
    "node_types": ["Sanction", "Requirement"]
  },
  "text": "Full training text with adequate context..."
}
```

## Metadata Fields

| Field | Required | Values | Description |
|-------|----------|--------|-------------|
| `section_type` | Yes | `"article"`, `"addenda"`, `"recitals"`, `"annex"`, `"guideline"` | Structural type of the source. |
| `article_ref` | Recommended | e.g. `"Article 5"`, `"Addenda Article 1"`, `"ANNEX III"`, `"Recital 12"` | Specific article/annex/recital identifier for grounding. |
| `node_types` | For relations | `["Sanction"]`, `["Requirement"]`, or both | Node types from which text was extracted (relations JSON only). |
| `domain` | For guidelines | Domain name if applicable | e.g. `"Healthcare"`, `"Employment"`, `"Essential Private Services"`. |
| `guideline_source` | For guidelines | e.g. `"PIPC"`, `"FSC"`, `"EU Commission"`, `"BSA"` | Issuing body of the guideline. |

## Text Content Constraints

1. **Adequate length**: Each `text` field should be long enough for meaningful context (typically 100–2000+ tokens). Shorter excerpts may be concatenated with surrounding context.
2. **Article tagging**: Where possible, include the article reference in or near the text (e.g., header line) for grounding.

## Source-Specific Rules

### Relations JSON (AIAct, KRAILaw)

- **section_type**:
  - AI Act: `"article"` (articles), `"recitals"` (rationale), `"annex"` (annexes)
  - Korea AI Law: `"article"` (main articles), `"addenda"` (Addenda Article N)
- **Text**: Use only `description` from relations where `target_node_type` or `start_node_type` is `Sanction` or `Requirement`, and only when the original text contains **"high-risk AI"**.
- **article_ref**: Derive from parent item `id` (e.g., `EU AI Act::Article 71` → `"Article 71"`).

### Guidelines (PDF)

- **section_type**: Use `"guideline"` or `"article"` when the doc explicitly covers a specific article.
- **Text**: Extract sections that mention Domain keywords (Healthcare, Education, Employment, etc.) or that cover a specific article. Split only on sentence boundaries so that no sentence is cut in the middle.
- **domain**: Populate when the section clearly pertains to a single Domain.
- **article_ref**: Populate when the section references a specific article (e.g., "Article 5", "제5조").
