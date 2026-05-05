# graphdb.py
from neo4j import GraphDatabase
import os
import json 


class Neo4jGraphDB:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(
                os.getenv("NEO4J_USER", "neo4j"),
                os.getenv("NEO4J_PASSWORD", "password"),
            ),
        )

    def close(self):
        self.driver.close()

    def store_graph(self, graph: dict) -> str:
        """Upsert all nodes and edges into Neo4j."""
        with self.driver.session() as session:
            # Nodes
            for entity_id, meta in graph["nodes"].items():
                session.run(
                    """
                    MERGE (n {entity_id: $entity_id})
                    SET n.type   = $type,
                        n.label  = $label,
                        n.authors = $authors,
                        n.link   = $link
                    """,
                    entity_id=entity_id,
                    type=meta.get("type", "unknown"),
                    label=meta.get("label", ""),
                    authors=meta.get("authors", []),
                    link=meta.get("link", ""),
                )

            # Edges
            for edge in graph["edges"]:
                session.run(
                    """
                    MATCH (a {entity_id: $source})
                    MATCH (b {entity_id: $target})
                    MERGE (a)-[r:RELATES {relation: $relation}]->(b)
                    SET r.evidence = $evidence
                    """,
                    source=edge["source"],
                    target=edge["target"],
                    relation=edge["relation"],
                    evidence=edge.get("evidence", ""),
                )

        return "Graph successfully stored in Neo4j."

    def query_graph(self, concept: str) -> str:
        """Retrieve subgraph around a concept — used by the analyst agent."""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (n {label: $concept})-[r]-(neighbor)
                RETURN n.label      AS node,
                       type(r)      AS relation,
                       r.relation   AS rel_type,
                       neighbor.label AS neighbor,
                       neighbor.type  AS neighbor_type,
                       r.evidence   AS evidence
                LIMIT 50
                """,
                concept=concept,
            )
            rows = [dict(record) for record in result]
            return json.dumps(rows, indent=2) if rows else f"No subgraph found for: {concept}"

    def get_full_graph_summary(self) -> str:
        """Returns node/edge counts + concept list for analyst context."""
        with self.driver.session() as session:
            counts = session.run(
                "MATCH (n) RETURN n.type AS type, count(*) AS count"
            )
            concepts = session.run(
                "MATCH (n {type: 'concept'}) RETURN n.label AS label"
            )
            summary = {
                "counts": [dict(r) for r in counts],
                "concepts": [r["label"] for r in concepts],
            }
            return json.dumps(summary, indent=2)