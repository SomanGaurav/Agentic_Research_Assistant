"""
=============================================================================
LAYER 2: GRAPH CONSTRUCTION LAYER
=============================================================================
Responsibilities:
  - Accept chunks from the Ingestion Layer (no direct PDF access)
  - For each chunk, call Claude API with a structured JSON schema prompt
    to extract:
      * Entities  (name, type, description)
      * Relations (source_entity, relation_type, target_entity, context)
  - Build an in-memory NetworkX directed graph
  - Expose query/traversal helpers for the Retrieval Layer

Graph Model:
  - Nodes represent entities: {id, name, type, description, papers, sections}
  - Edges represent relationships: {relation, context, chunk_id, paper_name}

Entity Types:
  MODEL, DATASET, METRIC, METHOD, CONCEPT, TASK, TOOL, PAPER, OTHER

Relation Types (examples):
  "improves", "uses", "evaluated_on", "compares_to", "based_on",
  "achieves", "outperforms", "proposes", "introduces", "trains_on"
=============================================================================
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import anthropic
# Would need to replace it with actual LLM used to extract knowledge graph from the chunks. Commented
# for now to avoid import errors . 
# Use gemini clients after understanding the graph builder . 
from ingestor import Chunk


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    name:        str
    type:        str   # MODEL | DATASET | METRIC | METHOD | CONCEPT | TASK | OTHER
    description: str
    papers:      List[str] = field(default_factory=list)
    sections:    List[str] = field(default_factory=list)

    @property
    def node_id(self) -> str:
        """Canonical node key: lowercased name for deduplication."""
        return self.name.lower().strip()


@dataclass
class Relation:
    source:     str   # entity node_id
    relation:   str   # e.g. "improves", "evaluated_on"
    target:     str   # entity node_id
    context:    str   # short sentence explaining the relationship
    chunk_id:   str
    paper_name: str


# ---------------------------------------------------------------------------
# JSON Schema for Claude structured extraction
# ---------------------------------------------------------------------------

EXTRACTION_SCHEMA = """
{
  "entities": [
    {
      "name": "string — the entity name exactly as it appears",
      "type": "one of: MODEL | DATASET | METRIC | METHOD | CONCEPT | TASK | TOOL | PAPER | OTHER",
      "description": "1-2 sentence description of this entity"
    }
  ],
  "relations": [
    {
      "source": "entity name (must appear in entities list above)",
      "relation": "short verb phrase, e.g. improves / uses / evaluated_on / outperforms / proposes / based_on / achieves / trains_on / compares_to / introduces",
      "target": "entity name (must appear in entities list above)",
      "context": "the sentence or short passage that expresses this relation"
    }
  ]
}
"""

EXTRACTION_SYSTEM = """You are a scientific knowledge graph extractor.
Given a chunk of text from a research paper, extract entities and relationships.

Return ONLY valid JSON matching this exact schema — no preamble, no markdown fences:
""" + EXTRACTION_SCHEMA


def _call_claude_extraction(text: str, client: anthropic.Anthropic) -> dict:
    """
    Call Claude to extract entities and relations from `text`.
    Returns parsed dict or empty scaffold on failure.
    """
    prompt = f"Extract entities and relationships from this research paper chunk:\n\n{text[:3000]}"

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        system=EXTRACTION_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()

    # Strip any accidental markdown fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Attempt to salvage partial JSON
        try:
            # Find the outermost { ... }
            start = raw.index("{")
            end   = raw.rindex("}") + 1
            return json.loads(raw[start:end])
        except Exception:
            return {"entities": [], "relations": []}


# ---------------------------------------------------------------------------
# Graph Builder
# ---------------------------------------------------------------------------

class KnowledgeGraphBuilder:
    """
    Builds a directed knowledge graph from ingested Chunk objects.

    Usage:
        builder = KnowledgeGraphBuilder()
        graph   = builder.build(chunks)
    """

    def __init__(self, api_client: Optional[anthropic.Anthropic] = None,
                 delay_between_calls: float = 0.3):
        self.client = api_client or anthropic.Anthropic()
        self.delay  = delay_between_calls
        self.graph: nx.DiGraph = nx.DiGraph()
        self._entities: Dict[str, Entity] = {}   # node_id → Entity
        self._relations: List[Relation]   = []

    # ------------------------------------------------------------------
    def build(self, chunks: List[Chunk]) -> "KnowledgeGraph":
        """
        Process all chunks, extract entities/relations, and return a
        KnowledgeGraph wrapper ready for retrieval queries.
        """
        total = len(chunks)
        print(f"\n  [GraphBuilder] Extracting entities from {total} chunks…")

        for i, chunk in enumerate(chunks, 1):
            print(f"    [{i}/{total}] chunk {chunk.chunk_id[:8]}… "
                  f"({chunk.paper_name} / {chunk.section})")

            result = _call_claude_extraction(chunk.text, self.client)
            self._ingest_extraction(result, chunk)

            if self.delay:
                time.sleep(self.delay)

        self._build_nx_graph()

        kg = KnowledgeGraph(
            graph     = self.graph,
            entities  = self._entities,
            relations = self._relations,
        )
        print(f"\n  [GraphBuilder] Graph built: "
              f"{kg.num_nodes} nodes, {kg.num_edges} edges")
        return kg

    # ------------------------------------------------------------------
    def _ingest_extraction(self, result: dict, chunk: Chunk) -> None:
        """Merge extracted entities and relations into internal stores."""
        raw_entities: List[dict] = result.get("entities", [])
        raw_relations: List[dict] = result.get("relations", [])

        # --- Entities ---
        seen_in_chunk: Dict[str, str] = {}  # original_name → node_id

        for ent_dict in raw_entities:
            name = ent_dict.get("name", "").strip()
            if not name or len(name) < 2:
                continue

            etype = ent_dict.get("type", "OTHER").upper()
            if etype not in {"MODEL", "DATASET", "METRIC", "METHOD",
                             "CONCEPT", "TASK", "TOOL", "PAPER", "OTHER"}:
                etype = "OTHER"

            desc = ent_dict.get("description", "")

            entity = Entity(
                name        = name,
                type        = etype,
                description = desc,
                papers      = [chunk.paper_name],
                sections    = [chunk.section],
            )
            nid = entity.node_id
            seen_in_chunk[name.lower()] = nid

            if nid in self._entities:
                # Merge: accumulate provenance
                existing = self._entities[nid]
                if chunk.paper_name not in existing.papers:
                    existing.papers.append(chunk.paper_name)
                if chunk.section not in existing.sections:
                    existing.sections.append(chunk.section)
                # Update description if the new one is longer
                if len(desc) > len(existing.description):
                    existing.description = desc
            else:
                self._entities[nid] = entity

        # --- Relations ---
        for rel_dict in raw_relations:
            src_name = rel_dict.get("source", "").strip().lower()
            tgt_name = rel_dict.get("target", "").strip().lower()
            rel_type = rel_dict.get("relation", "").strip().lower()
            context  = rel_dict.get("context", "").strip()

            # Both endpoints must have been extracted in this call or
            # already exist in the global entity store
            src_id = seen_in_chunk.get(src_name) or src_name
            tgt_id = seen_in_chunk.get(tgt_name) or tgt_name

            if not src_id or not tgt_id or not rel_type:
                continue

            relation = Relation(
                source     = src_id,
                relation   = rel_type,
                target     = tgt_id,
                context    = context,
                chunk_id   = chunk.chunk_id,
                paper_name = chunk.paper_name,
            )
            self._relations.append(relation)

    # ------------------------------------------------------------------
    def _build_nx_graph(self) -> None:
        """Materialise entities and relations as a NetworkX DiGraph."""
        self.graph.clear()

        for nid, entity in self._entities.items():
            self.graph.add_node(nid, **{
                "name":        entity.name,
                "type":        entity.type,
                "description": entity.description,
                "papers":      entity.papers,
                "sections":    entity.sections,
            })

        for rel in self._relations:
            # Ensure both endpoints exist (relations may reference entities
            # extracted in other chunks)
            for endpoint in (rel.source, rel.target):
                if not self.graph.has_node(endpoint):
                    self.graph.add_node(endpoint, name=endpoint,
                                        type="OTHER", description="",
                                        papers=[], sections=[])

            self.graph.add_edge(
                rel.source, rel.target,
                relation   = rel.relation,
                context    = rel.context,
                chunk_id   = rel.chunk_id,
                paper_name = rel.paper_name,
            )


# ---------------------------------------------------------------------------
# Knowledge Graph — Query Interface
# ---------------------------------------------------------------------------

class KnowledgeGraph:
    """
    Thin wrapper around a NetworkX DiGraph that exposes the query API
    consumed by the Retrieval Layer.

    Public methods:
        search_entities(keyword)             → List[dict]
        get_neighbours(node_id, depth)       → List[dict]
        get_entity_context(node_id)          → dict
        find_path(src, tgt)                  → List[str]
        subgraph_for_query(keywords)         → List[dict]
        to_serialisable()                    → dict
    """

    def __init__(self, graph: nx.DiGraph,
                 entities: Dict[str, Entity],
                 relations: List[Relation]):
        self.graph     = graph
        self.entities  = entities
        self.relations = relations

    @property
    def num_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self.graph.number_of_edges()

    # ------------------------------------------------------------------
    def search_entities(self, keyword: str, top_k: int = 10) -> List[dict]:
        """
        Return entities whose name or description contains `keyword`.
        Results are ranked: exact match > name prefix > substring > description.
        """
        kw = keyword.lower()
        scored: List[Tuple[int, dict]] = []

        for nid, data in self.graph.nodes(data=True):
            name = data.get("name", "").lower()
            desc = data.get("description", "").lower()

            if name == kw:
                score = 100
            elif name.startswith(kw):
                score = 80
            elif kw in name:
                score = 60
            elif kw in desc:
                score = 30
            else:
                continue

            scored.append((score, {"node_id": nid, **data}))

        scored.sort(key=lambda x: -x[0])
        return [item for _, item in scored[:top_k]]

    # ------------------------------------------------------------------
    def get_neighbours(self, node_id: str, depth: int = 1) -> List[dict]:
        """
        Return all nodes reachable within `depth` hops from `node_id`,
        including the connecting edges.
        """
        if not self.graph.has_node(node_id):
            return []

        results = []
        visited = {node_id}
        frontier = [node_id]

        for _ in range(depth):
            next_frontier = []
            for nid in frontier:
                for successor in self.graph.successors(nid):
                    if successor not in visited:
                        edge_data = self.graph.get_edge_data(nid, successor)
                        results.append({
                            "from":      nid,
                            "to":        successor,
                            "relation":  edge_data.get("relation", ""),
                            "context":   edge_data.get("context",  ""),
                            "node_data": dict(self.graph.nodes[successor]),
                        })
                        visited.add(successor)
                        next_frontier.append(successor)

                for predecessor in self.graph.predecessors(nid):
                    if predecessor not in visited:
                        edge_data = self.graph.get_edge_data(predecessor, nid)
                        results.append({
                            "from":      predecessor,
                            "to":        nid,
                            "relation":  edge_data.get("relation", ""),
                            "context":   edge_data.get("context",  ""),
                            "node_data": dict(self.graph.nodes[predecessor]),
                        })
                        visited.add(predecessor)
                        next_frontier.append(predecessor)

            frontier = next_frontier

        return results

    # ------------------------------------------------------------------
    def get_entity_context(self, node_id: str) -> dict:
        """Return full node data plus all incident edges."""
        if not self.graph.has_node(node_id):
            return {}

        outgoing = [
            {
                "direction": "out",
                "neighbour": tgt,
                **self.graph.get_edge_data(node_id, tgt),
            }
            for tgt in self.graph.successors(node_id)
        ]
        incoming = [
            {
                "direction": "in",
                "neighbour": src,
                **self.graph.get_edge_data(src, node_id),
            }
            for src in self.graph.predecessors(node_id)
        ]

        return {
            "node":     dict(self.graph.nodes[node_id]),
            "outgoing": outgoing,
            "incoming": incoming,
        }

    # ------------------------------------------------------------------
    def find_path(self, source: str, target: str) -> List[str]:
        """Return the shortest path between two entity node IDs."""
        try:
            return nx.shortest_path(self.graph.to_undirected(), source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    # ------------------------------------------------------------------
    def subgraph_for_query(self, keywords: List[str],
                           neighbour_depth: int = 1) -> List[dict]:
        """
        Given a list of query keywords, find matching entities and expand
        each by `neighbour_depth` hops. Returns a flat list of edge dicts
        suitable for context injection into prompts.
        """
        seed_nodes = set()
        for kw in keywords:
            for ent in self.search_entities(kw, top_k=3):
                seed_nodes.add(ent["node_id"])

        all_edges: List[dict] = []
        seen_edges: set = set()

        for nid in seed_nodes:
            for edge in self.get_neighbours(nid, depth=neighbour_depth):
                key = (edge["from"], edge["to"], edge["relation"])
                if key not in seen_edges:
                    all_edges.append(edge)
                    seen_edges.add(key)

        return all_edges

    # ------------------------------------------------------------------
    def to_serialisable(self) -> dict:
        """Serialise graph to a JSON-friendly dict (for persistence/debug)."""
        return {
            "nodes": [
                {"node_id": nid, **data}
                for nid, data in self.graph.nodes(data=True)
            ],
            "edges": [
                {"from": u, "to": v, **data}
                for u, v, data in self.graph.edges(data=True)
            ],
        }