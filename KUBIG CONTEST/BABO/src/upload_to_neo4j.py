"""
Import relations_ready.json into Neo4j per Ling Long Protocol (relations_extraction.md).

Schema-aligned nodes: Article, Rationale, Annex, Chunk, Regulation; Concept, Stakeholder,
Domain, Requirement, Sanction, Support, TechCriterion, UsageCriterion, RiskCategory.
Relationships: DEFINES, IMPOSES, MANDATED_FOR, PERMITS, APPLIES_TO, DETAILS_DEFINITION_OF,
DETAILS_SCOPE, ESTABLISHES, LEADS_TO, TRIGGERS, PENALIZES_WITH, IS_A, ENCOMPASSES, SUPPLEMENTS, INCLUDES.
§3.4 Node Centered: description in Node properties, NOT on relationships. §3.1 (first general constraint):
(Sanction) MUST be connected to (Stakeholder) via [:APPLIES_TO]. MANDATED_FOR/PERMITS chain
from Requirement (§3.4–3.5). APPLIES_TO: Sanction -> Stakeholder (§2). Node Centered (§3.4):
text in node properties (Sanction.description, Requirement.description, TechCriterion.detail).
Node Integrity (§3.5): fill node properties as much as possible. Relevance (§3.6): descriptions
are directly relevant excerpts, not full-text.

Loads NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD from .env.
"""

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _PROJECT_ROOT / ".env"
_JSON_PATH = _PROJECT_ROOT / "data" / "processed" / "relations_ready.json"

# MANDATED_FOR and PERMITS chain from Requirement (logical start), not Article (§3.4–3.5)
CHAIN_FROM_REQUIREMENT = {"MANDATED_FOR", "PERMITS"}
# APPLIES_TO: Sanction -> Stakeholder (§2); requires start_node_type=Sanction


def _lang_from_source(source: str) -> str:
    """Infer Concept.lang from source (schema §1)."""
    return "en" if "EU" in source or "Europe" in source else "ko"


def get_container_type(container_id: str) -> str:
    """Map container id to Neo4j label."""
    if "Recital" in container_id or "recital" in container_id.lower():
        return "Rationale"
    if "ANNEX" in container_id.upper() or "Annex" in container_id:
        return "Annex"
    return "Article"


def safe_rel_type(rel_type: str) -> str:
    """Convert relation type to valid Cypher relationship type (uppercase, no special chars)."""
    s = re.sub(r"[^A-Za-z0-9_]", "_", rel_type).upper()
    return s if s else "RELATES_TO"


def process_block(
    driver, block: dict, source: str, block_key: str
) -> tuple[int, int]:
    """Process eu_ai_act or korea_ai_law block. Returns (nodes_created, rels_created)."""
    nodes, rels = 0, 0

    def run(query: str, params: dict | None = None) -> None:
        nonlocal nodes, rels
        with driver.session() as session:
            result = session.run(query, params or {})
            for record in result:
                if "nodes" in record:
                    nodes += record["nodes"]
                if "rels" in record:
                    rels += record["rels"]

    for section, label in [
        ("rationale", "Rationale"),
        ("articles", "Article"),
        ("annexes", "Annex"),
    ]:
        items = block.get(section) or []
        for item in items:
            cid = item.get("id") or ""
            full_text = (item.get("full_text") or "")[:50000]  # limit size
            metadata = item.get("metadata") or {}
            meta_json = json.dumps(metadata) if metadata else "{}"
            source_file = item.get("source_file") or ""
            updated_at = item.get("updated_at") or ""
            title = item.get("title") or metadata.get("title") or ""

            run(
                f"""
                MERGE (c:{label} {{id: $id}})
                SET c.full_text = $full_text, c.source = $source, c.metadata = $metadata,
                    c.source_file = $source_file, c.updated_at = $updated_at, c.title = $title
                """,
                {
                    "id": cid,
                    "full_text": full_text,
                    "source": source,
                    "metadata": meta_json,
                    "source_file": source_file,
                    "updated_at": updated_at,
                    "title": title,
                },
            )
            nodes += 1

            chunks = item.get("chunks") or []
            for ch in chunks:
                chunk_id = ch.get("chunk_id") or ""
                text = (ch.get("text") or "")[:100000]
                emb = ch.get("embedding")
                if not chunk_id or not text:
                    continue
                emb_param = emb if emb else []
                full_chunk_id = f"{source}::{chunk_id}"
                run(
                    f"""
                    MERGE (ch:Chunk {{chunk_id: $chunk_id}})
                    SET ch.text = $text, ch.embedding = $embedding
                    WITH ch
                    MATCH (c:{label} {{id: $container_id}})
                    MERGE (c)-[:HAS_CHUNK]->(ch)
                    """,
                    {
                        "chunk_id": full_chunk_id,
                        "text": text,
                        "embedding": emb_param,
                        "container_id": cid,
                    },
                )
                nodes += 1
                rels += 1

            current_requirement: str | None = None
            relations = item.get("relations") or []
            src_file = item.get("source_file") or ""
            upd_at = item.get("updated_at") or ""
            for rel in relations:
                rtype = rel.get("type") or ""
                target_type = rel.get("target_node_type") or ""
                target_name = rel.get("target_node_name") or ""
                desc = (rel.get("description") or "")[:10000]
                start_type = rel.get("start_node_type")
                start_name = rel.get("start_node_name")

                if not target_name:
                    continue

                rel_label = safe_rel_type(rtype)
                # RiskCategory uses level (schema §1), not name
                target_prop = "level" if target_type == "RiskCategory" else "name"
                lang = _lang_from_source(source) if target_type == "Concept" else None

                if rtype == "IMPOSES":
                    current_requirement = target_name

                if rtype in CHAIN_FROM_REQUIREMENT and current_requirement:
                    run(
                        f"""
                        MERGE (req:Requirement {{name: $req_name}})
                        ON CREATE SET req.source_file = $src_file, req.updated_at = $upd_at
                        MERGE (t:{target_type} {{{target_prop}: $target_val}})
                        ON CREATE SET t.source_file = $src_file, t.updated_at = $upd_at
                        MERGE (req)-[r:{rel_label}]->(t)
                        """,
                        {
                            "req_name": current_requirement,
                            "target_val": target_name,
                            "desc": desc,
                            "src_file": src_file,
                            "upd_at": upd_at,
                        },
                    )
                    nodes += 2
                    rels += 1
                elif start_type and start_name:
                    s_prop = "id" if start_type in ("Article", "Annex", "Rationale") else "name"
                    t_prop = "level" if target_type == "RiskCategory" else "name"
                    onCreate = " ON CREATE SET t.lang = $lang" if target_type == "Concept" and lang else ""
                    # Node Centered (§3.4): text in node props; Node Integrity (§3.5): fill props
                    if rtype == "IMPOSES" and target_type == "Requirement":
                        node_extra = " SET t.description = $desc, t.source_file = $src_file, t.updated_at = $upd_at"
                    elif rtype == "PENALIZES_WITH" and target_type == "Sanction":
                        node_extra = " SET t.description = $desc, t.source_file = $src_file, t.updated_at = $upd_at"
                    elif rtype == "ESTABLISHES" and target_type in ("TechCriterion", "UsageCriterion"):
                        node_extra = " SET t.detail = $desc, t.source_file = $src_file, t.updated_at = $upd_at"
                    else:
                        node_extra = " ON CREATE SET t.source_file = $src_file, t.updated_at = $upd_at"
                    params: dict = {"start_name": start_name, "target_val": target_name, "desc": desc, "src_file": src_file, "upd_at": upd_at}
                    if lang is not None:
                        params["lang"] = lang
                    run(
                        f"""
                        MERGE (s:{start_type} {{{s_prop}: $start_name}})
                        MERGE (t:{target_type} {{{t_prop}: $target_val}}){onCreate}{node_extra}
                        MERGE (s)-[r:{rel_label}]->(t)
                        """,
                        params,
                    )
                    nodes += 2
                    rels += 1
                else:
                    onCreate = " ON CREATE SET t.lang = $lang" if target_type == "Concept" and lang else ""
                    # Node Centered (§3.4): text in node; Node Integrity (§3.5): fill props
                    node_extra = " SET t.description = $desc, t.source_file = $src_file, t.updated_at = $upd_at"
                    if target_type in ("TechCriterion", "UsageCriterion"):
                        node_extra = " SET t.detail = $desc, t.source_file = $src_file, t.updated_at = $upd_at"
                    params: dict = {"container_id": cid, "target_val": target_name, "desc": desc, "src_file": src_file, "upd_at": upd_at}
                    if lang is not None:
                        params["lang"] = lang
                    run(
                        f"""
                        MATCH (c:{label} {{id: $container_id}})
                        MERGE (t:{target_type} {{{target_prop}: $target_val}}){onCreate}{node_extra}
                        MERGE (c)-[r:{rel_label}]->(t)
                        """,
                        params,
                    )
                    nodes += 1
                    rels += 1

    return nodes, rels


def process_regulation_and_includes(
    driver, block: dict, source: str
) -> tuple[int, int]:
    """Create Regulation node and INCLUDES edges to Article/Rationale/Annex (schema §2)."""
    nodes, rels = 0, 0
    reg = block.get("regulation") or {}
    reg_name = reg.get("name") or source
    jurisdiction = reg.get("jurisdiction") or ""
    effective_date = reg.get("effective_date") or ""

    if not reg_name:
        return 0, 0

    with driver.session() as session:
        session.run(
            """
            MERGE (r:Regulation {name: $name})
            SET r.jurisdiction = $jurisdiction, r.effective_date = $effective_date
            """,
            {"name": reg_name, "jurisdiction": jurisdiction, "effective_date": effective_date},
        )
    nodes += 1

    for section, label in [
        ("rationale", "Rationale"),
        ("articles", "Article"),
        ("annexes", "Annex"),
    ]:
        for item in (block.get(section) or []):
            cid = item.get("id")
            if not cid:
                continue
            with driver.session() as session:
                session.run(
                    f"""
                    MATCH (r:Regulation {{name: $reg_name}})
                    MATCH (c:{label} {{id: $container_id}})
                    MERGE (r)-[:INCLUDES]->(c)
                    """,
                    {"reg_name": reg_name, "container_id": cid},
                )
            rels += 1

    return nodes, rels


def process_cross_jurisdictional(driver, relations: list) -> tuple[int, int]:
    """Process cross_jurisdictional_relations. Returns (nodes, rels)."""
    nodes, rels = 0, 0

    for rel in relations or []:
        rtype = rel.get("type") or ""
        start_type = rel.get("start_node_type") or "Concept"
        start_name = rel.get("start_node_name") or ""
        target_type = rel.get("target_node_type") or "Concept"
        target_name = rel.get("target_node_name") or ""
        desc = (rel.get("description") or "")[:10000]

        if not start_name or not target_name:
            continue

        rel_label = safe_rel_type(rtype)
        # Start: Article/Annex/Rationale use id; Concept uses name
        if start_type in ("Article", "Annex", "Rationale"):
            start_prop, start_val = "id", start_name
        else:
            start_prop, start_val = "name", start_name
        # Target: RiskCategory uses level; others use name
        target_prop = "level" if target_type == "RiskCategory" else "name"
        # Concept.lang: ENCOMPASSES = start en, end ko; SUPPLEMENTS = target from other jurisdiction
        start_lang = end_lang = None
        if rtype == "ENCOMPASSES":
            start_lang, end_lang = "en", "ko"
        elif rtype == "SUPPLEMENTS" and target_type == "Concept":
            end_lang = "ko" if "EU" in start_name else "en"

        s_on_create = f" ON CREATE SET s.lang = $start_lang" if start_type == "Concept" and start_lang else ""
        t_on_create = f" ON CREATE SET t.lang = $end_lang" if target_type == "Concept" and end_lang else ""
        params: dict = {"start_val": start_val, "target_val": target_name, "desc": desc}
        if start_lang is not None:
            params["start_lang"] = start_lang
        if end_lang is not None:
            params["end_lang"] = end_lang

        with driver.session() as session:
            session.run(
                f"""
                MERGE (s:{start_type} {{{start_prop}: $start_val}}){s_on_create}
                MERGE (t:{target_type} {{{target_prop}: $target_val}}){t_on_create}
                MERGE (s)-[r:{rel_label}]->(t)
                """,
                params,
            )
        nodes += 2
        rels += 1

    return nodes, rels


def _validate_sanctions_have_applies_to(driver) -> None:
    """§3.1: Every (Sanction) MUST be connected to (Stakeholder) via [:APPLIES_TO]."""
    with driver.session() as session:
        result = session.run(
            """
            MATCH (s:Sanction)
            WHERE NOT (s)-[:APPLIES_TO]->(:Stakeholder)
            RETURN s.name AS name
            """
        )
        orphans = [r["name"] for r in result if r["name"]]
    if orphans:
        raise SystemExit(
            f"Schema violation (§3.1): Sanction(s) must be connected to Stakeholder via [:APPLIES_TO]. "
            f"Sanctions without APPLIES_TO: {orphans}"
        )


def create_vector_index(driver) -> None:
    """Create vector index on Chunk.embedding if supported (Neo4j 5.11+)."""
    with driver.session() as session:
        try:
            session.run(
                """
                CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
                FOR (c:Chunk) ON (c.embedding)
                OPTIONS {indexConfig: {
                    `vector.dimensions`: 1536,
                    `vector.similarity_function`: 'cosine'
                }}
                """
            )
            print("  Created vector index on Chunk.embedding")
        except Exception as e:
            print(f"  Skipped vector index (may need Neo4j 5.11+): {e}")


def main() -> None:
    load_dotenv(_ENV_PATH)
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    if not all([uri, user, password]):
        raise SystemExit("NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD required in .env")

    print(f"Loading {_JSON_PATH}...")
    with open(_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    driver = GraphDatabase.driver(uri, auth=(user, password))
    total_nodes, total_rels = 0, 0

    try:
        # Clear existing (optional - comment out to append)
        # with driver.session() as s:
        #     s.run("MATCH (n) DETACH DELETE n")

        eu = data.get("eu_ai_act") or {}
        kr = data.get("korea_ai_law") or {}
        cross = data.get("cross_jurisdictional_relations") or []

        print("Importing EU AI Act...")
        n, r = process_block(driver, eu, eu.get("source", "EU AI Act"), "eu_ai_act")
        total_nodes += n
        total_rels += r
        print(f"  EU: ~{n} nodes, ~{r} relationships")
        n2, r2 = process_regulation_and_includes(driver, eu, "EU AI Act")
        total_nodes += n2
        total_rels += r2
        if n2 or r2:
            print(f"  EU Regulation + INCLUDES: ~{n2} nodes, ~{r2} relationships")

        print("Importing Korea AI Law...")
        n, r = process_block(driver, kr, kr.get("source", "Korea AI Law"), "korea_ai_law")
        total_nodes += n
        total_rels += r
        print(f"  KR: ~{n} nodes, ~{r} relationships")
        n2, r2 = process_regulation_and_includes(driver, kr, "Korea AI Law")
        total_nodes += n2
        total_rels += r2
        if n2 or r2:
            print(f"  KR Regulation + INCLUDES: ~{n2} nodes, ~{r2} relationships")

        if cross:
            print("Importing cross-jurisdictional relations...")
            n, r = process_cross_jurisdictional(driver, cross)
            total_nodes += n
            total_rels += r
            print(f"  Cross: ~{n} nodes, ~{r} relationships")

        print("Creating vector index...")
        create_vector_index(driver)

        # §3.1 (first general constraint): (Sanction) MUST be connected to (Stakeholder) via [:APPLIES_TO]
        _validate_sanctions_have_applies_to(driver)

        print(f"\nDone. Total: ~{total_nodes} nodes, ~{total_rels} relationships.")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
