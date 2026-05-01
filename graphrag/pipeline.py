"""
=============================================================================
INTEGRATED PIPELINE — GraphRAG Research Paper Analysis System
=============================================================================
This module wires all four layers together into a single pipeline object.

Architecture:
  [PDFs] → Ingestion → [Chunks]
                          ↓
                   Graph Construction → [KnowledgeGraph]
                          ↓                     ↓
                   VectorIndex ────────── HybridRetriever
                                                ↓
                                       [RetrievalResult]
                                                ↓
                                     AgentOrchestrator
                                                ↓
                                      [AgentResponse(s)]

Usage:
    pipeline = GraphRAGPipeline(papers_dir="./papers")
    pipeline.build()                         # one-time setup
    responses = pipeline.query("How does X compare to Y?")
=============================================================================
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import anthropic

# Resolve imports relative to the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestor           import PaperIngestor, Chunk
from builder                import KnowledgeGraphBuilder, KnowledgeGraph
from retriever   import VectorIndex, HybridRetriever, RetrievalResult
from agents                import (
    AgentOrchestrator, AgentResponse, AgentType
)


class GraphRAGPipeline:
    """
    End-to-end pipeline that ingests papers, builds the graph, sets up
    retrieval, and exposes a simple `.query()` interface.
    """

    def __init__(
        self,
        papers_dir:           str = "./papers",
        max_words_per_chunk:  int = 400,
        api_key:              Optional[str] = None,
    ):
        self.papers_dir          = papers_dir
        self.max_words_per_chunk = max_words_per_chunk
        self.client              = anthropic.Anthropic(
            api_key=api_key) if api_key else anthropic.Anthropic()

        # State — populated after build()
        self.chunks:       List[Chunk]      = []
        self.kg:           Optional[KnowledgeGraph]  = None
        self.retriever:    Optional[HybridRetriever] = None
        self.orchestrator: Optional[AgentOrchestrator] = None
        self._built = False

    # ------------------------------------------------------------------
    def build(self) -> None:
        """
        Run all setup steps:
          1. Ingest PDFs → chunks
          2. Build knowledge graph from chunks
          3. Build vector index from chunks
          4. Wire HybridRetriever and AgentOrchestrator
        """
        t0 = time.time()
        print("\n" + "="*60)
        print("  GRAPHRAG PIPELINE — BUILD PHASE")
        print("="*60)

        # ---- Layer 1: Ingestion ----
        print("\n[STEP 1/4] Ingestion Layer")
        ingestor     = PaperIngestor(self.papers_dir, self.max_words_per_chunk)
        self.chunks  = ingestor.ingest_all()

        # ---- Layer 2: Graph Construction ----
        print("\n[STEP 2/4] Graph Construction Layer")
        builder  = KnowledgeGraphBuilder(api_client=self.client)
        self.kg  = builder.build(self.chunks)

        # ---- Layer 3: Vector Index ----
        print("\n[STEP 3/4] Vector Index (Retrieval Layer)")
        vi = VectorIndex()
        vi.build(self.chunks)
        self.retriever = HybridRetriever(vi, self.kg)

        # ---- Layer 4: Agent Orchestrator ----
        print("\n[STEP 4/4] Agent Orchestrator (Agent Layer)")
        self.orchestrator = AgentOrchestrator(client=self.client)
        self._built = True

        elapsed = time.time() - t0
        print(f"\n{'='*60}")
        print(f"  BUILD COMPLETE in {elapsed:.1f}s")
        print(f"  Chunks: {len(self.chunks)} | "
              f"Graph: {self.kg.num_nodes} nodes / {self.kg.num_edges} edges")
        print("="*60 + "\n")

    # ------------------------------------------------------------------
    def query(
        self,
        question: str,
        force_agents: Optional[List[AgentType]] = None,
    ) -> List[AgentResponse]:
        """
        Run an end-to-end query through the pipeline.

        Args:
            question     : natural language user query
            force_agents : optionally bypass router and specify agents

        Returns:
            list of AgentResponse objects
        """
        self._check_built()

        print(f"\n{'─'*60}")
        print(f"  QUERY: {question}")
        print(f"{'─'*60}")

        # ---- Retrieval ----
        print("\n[Retrieval] Running hybrid retrieval…")
        context = self.retriever.retrieve(question)
        print(f"  → {len(context.fused)} fused results")

        # ---- Agents ----
        print("\n[Agents] Invoking orchestrator…")
        responses = self.orchestrator.run(question, context, force_agents)

        return responses

    # ------------------------------------------------------------------
    def save_graph(self, output_path: str = "knowledge_graph.json") -> None:
        """Serialise the knowledge graph to a JSON file."""
        self._check_built()
        data = self.kg.to_serialisable()
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"  [Pipeline] Graph saved to {output_path}")

    # ------------------------------------------------------------------
    def _check_built(self) -> None:
        if not self._built:
            raise RuntimeError("Call pipeline.build() before pipeline.query()")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    """
    Example run demonstrating all four layers and three agent types.
    Usage: python pipeline/pipeline.py
    """
    import os

    # --- Setup ---
    papers_dir = Path(__file__).parent.parent / "papers"
    papers_dir.mkdir(exist_ok=True)

    if not list(papers_dir.glob("*.pdf")):
        print(f"\nNo PDFs found in {papers_dir}")
        print("Please add research paper PDFs to the 'papers/' directory and re-run.")
        print("\nTo demo with synthetic data, run: python pipeline/demo_run.py")
        return

    # --- Build pipeline ---
    pipeline = GraphRAGPipeline(papers_dir=str(papers_dir))
    pipeline.build()

    # --- Save graph for inspection ---
    pipeline.save_graph("knowledge_graph.json")

    # --- Example queries ---
    example_queries = [
        ("summary",    "What are the main contributions of the papers in this corpus?"),
        ("technical",  "Compare the methods and architectures used across the papers."),
        ("visual",     "Show the relationships between models, datasets, and evaluation metrics."),
        ("multi",      "Summarise and visualise how the key models relate to each other."),
    ]

    for label, query in example_queries:
        print(f"\n{'#'*60}")
        print(f"  EXAMPLE QUERY [{label.upper()}]")
        print(f"{'#'*60}")

        responses = pipeline.query(query)
        for resp in responses:
            print(resp)


if __name__ == "__main__":
    main()