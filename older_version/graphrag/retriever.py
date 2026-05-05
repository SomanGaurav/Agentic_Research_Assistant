"""
=============================================================================
LAYER 3: RETRIEVAL LAYER  —  Hybrid Graph RAG
=============================================================================
Responsibilities:
  - Vector-based retrieval: embed all chunks with a sentence-transformer,
    build a FAISS index, retrieve top-k chunks for a query
  - Graph-based retrieval: parse query keywords, search the KnowledgeGraph
    for matching entities and their neighbourhood edges
  - Fusion: merge both result streams, de-duplicate, and rank by a
    combined relevance score
  - Output: RetrievalResult — a unified context object consumed by agents

Query Processing Pipeline:
  1. Embed query → cosine search over FAISS → top-k vector hits
  2. Tokenise query → entity search in KG → neighbour expansion
  3. Merge: vector hits + graph context → deduplicated, ranked
  4. Return RetrievalResult with separate fields for transparency

Relevance Scoring (hybrid):
  - Vector score  : cosine similarity ∈ [0, 1]   (weight α = 0.6)
  - Graph score   : 1.0 for exact entity match,
                    0.7 for 1-hop neighbour,
                    0.4 for 2-hop neighbour       (weight β = 0.4)
  - Final score   : α * vector_score + β * graph_score
=============================================================================
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import math
from collections import Counter

import faiss
import numpy as np

# Attempt to import sentence-transformers; fall back to a TF-IDF embedder
# so the system runs in sandboxed / offline environments.
try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

from older_version.graphrag.ingestor import Chunk
from older_version.graphrag.builder import KnowledgeGraph


# ---------------------------------------------------------------------------
# TF-IDF Fallback Embedder (no external model required)
# ---------------------------------------------------------------------------

class _TFIDFEmbedder:
    """
    Lightweight TF-IDF vectoriser used when sentence-transformers is
    unavailable or the model cannot be downloaded (sandbox environments).

    Produces normalised sparse-to-dense projected vectors via a fixed
    random projection matrix so FAISS inner-product == cosine similarity.
    """
    DIM = 384

    def __init__(self):
        self.vocab:      Dict[str, int] = {}
        self.idf:        Dict[str, float] = {}
        self._proj:      Optional[np.ndarray] = None   # vocab_size × DIM
        self._fitted     = False

    def _tokenise(self, text: str) -> List[str]:
        return re.findall(r"[a-z][a-z0-9]{1,}", text.lower())

    def fit(self, texts: List[str]) -> None:
        """Build vocabulary and IDF weights."""
        N = len(texts)
        df: Counter = Counter()
        all_tokens: List[List[str]] = []
        for t in texts:
            toks = set(self._tokenise(t))
            all_tokens.append(list(toks))
            df.update(toks)

        # Build vocab
        for term, freq in df.most_common(8000):
            if term not in self.vocab:
                self.vocab[term] = len(self.vocab)

        # IDF = log((N+1)/(df+1)) + 1
        self.idf = {
            t: math.log((N + 1) / (f + 1)) + 1.0
            for t, f in df.items() if t in self.vocab
        }

        # Fixed random projection: vocab_size × DIM
        rng = np.random.default_rng(42)
        vocab_size = len(self.vocab)
        self._proj = rng.standard_normal((vocab_size, self.DIM)).astype(np.float32)
        self._fitted = True

    def encode(self, texts: List[str],
               normalize_embeddings: bool = True,
               batch_size: int = 32,
               show_progress_bar: bool = False) -> np.ndarray:
        if not self._fitted:
            self.fit(texts)

        embeddings = np.zeros((len(texts), self.DIM), dtype=np.float32)

        for i, text in enumerate(texts):
            toks = self._tokenise(text)
            tf: Counter = Counter(toks)
            total = max(len(toks), 1)
            sparse = np.zeros(len(self.vocab), dtype=np.float32)
            for term, count in tf.items():
                idx = self.vocab.get(term)
                if idx is not None:
                    sparse[idx] = (count / total) * self.idf.get(term, 1.0)
            # Project to DIM dimensions
            embeddings[i] = sparse @ self._proj

        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            embeddings /= norms

        return embeddings


# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

EMBED_MODEL      = "all-MiniLM-L6-v2"   # fast, 384-dim, production-quality
VECTOR_TOP_K     = 8                     # raw vector hits before fusion
GRAPH_TOP_K      = 5                     # max entity seeds from query
FUSION_TOP_N     = 10                    # final ranked results returned
ALPHA            = 0.6                   # weight for vector score
BETA             = 0.4                   # weight for graph score


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class VectorHit:
    chunk:          Chunk
    vector_score:   float   # cosine similarity


@dataclass
class GraphHit:
    node_id:     str
    name:        str
    entity_type: str
    description: str
    relations:   List[dict]   # neighbouring edges
    graph_score: float


@dataclass
class FusedResult:
    """A single ranked result after fusing vector + graph signals."""
    chunk:          Optional[Chunk]
    vector_score:   float
    graph_score:    float
    final_score:    float
    graph_hits:     List[GraphHit] = field(default_factory=list)


@dataclass
class RetrievalResult:
    """
    The unified context object passed to the Agent Layer.

    Fields:
        query          : original user query string
        fused          : top-N FusedResult objects, sorted by final_score
        graph_context  : human-readable string summarising graph facts
        vector_context : human-readable string of top chunk texts
        full_context   : combined context ready for LLM prompt injection
    """
    query:          str
    fused:          List[FusedResult]
    graph_context:  str
    vector_context: str
    full_context:   str


# ---------------------------------------------------------------------------
# Vector Index
# ---------------------------------------------------------------------------

class VectorIndex:
    """
    Wraps a FAISS flat inner-product index over chunk embeddings.
    Embeddings are L2-normalised so inner product == cosine similarity.

    Uses sentence-transformers when available; falls back to the built-in
    TF-IDF embedder for offline / sandboxed environments.
    """

    def __init__(self, model_name: str = EMBED_MODEL):
        if _ST_AVAILABLE:
            try:
                print(f"  [VectorIndex] Loading embedding model: {model_name}")
                self.model = _SentenceTransformer(model_name)
                self._using_tfidf = False
            except Exception:
                print("  [VectorIndex] Model download failed — using TF-IDF fallback")
                self.model = _TFIDFEmbedder()
                self._using_tfidf = True
        else:
            print("  [VectorIndex] sentence-transformers unavailable — using TF-IDF fallback")
            self.model = _TFIDFEmbedder()
            self._using_tfidf = True

        self.chunks: List[Chunk] = []
        self.index:  Optional[faiss.IndexFlatIP] = None

    # ------------------------------------------------------------------
    def build(self, chunks: List[Chunk]) -> None:
        """Embed all chunks and build the FAISS index."""
        print(f"  [VectorIndex] Embedding {len(chunks)} chunks...")
        texts = [c.text for c in chunks]

        # Fit TF-IDF on corpus if using fallback
        if self._using_tfidf:
            self.model.fit(texts)

        embeddings = self.model.encode(
            texts,
            batch_size    = 32,
            show_progress_bar = False,
            normalize_embeddings = True,   # L2 normalise -> cosine via IP
        ).astype(np.float32)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.chunks = chunks
        print(f"  [VectorIndex] Index built: {self.index.ntotal} vectors ({dim}d)")

    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = VECTOR_TOP_K) -> List[VectorHit]:
        """Return top-k chunks most similar to `query`."""
        if self.index is None:
            raise RuntimeError("VectorIndex.build() must be called before search()")

        q_emb = self.model.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)

        k      = min(top_k, len(self.chunks))
        scores, indices = self.index.search(q_emb, k)

        hits = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            hits.append(VectorHit(
                chunk        = self.chunks[idx],
                vector_score = float(score),
            ))
        return hits


# ---------------------------------------------------------------------------
# Graph Retriever
# ---------------------------------------------------------------------------

# Simple English stopwords to filter from query tokens
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "and", "or",
    "but", "not", "this", "that", "it", "its", "as", "do", "does", "did",
    "what", "how", "why", "which", "who", "when", "where", "can", "could",
    "would", "should", "will", "have", "has", "had", "about", "than", "into",
}

def _tokenise_query(query: str) -> List[str]:
    """Extract meaningful tokens from a query string."""
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-_]{1,}", query)
    return [t for t in tokens if t.lower() not in _STOPWORDS and len(t) >= 3]


class GraphRetriever:
    """
    Retrieves graph context (entities + neighbourhood) relevant to a query.
    """

    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg = knowledge_graph

    # ------------------------------------------------------------------
    def search(self, query: str,
               top_k_entities: int = GRAPH_TOP_K,
               neighbour_depth: int = 1) -> List[GraphHit]:
        """
        1. Tokenise query
        2. Search entities by each keyword
        3. Expand each matched entity by `neighbour_depth` hops
        4. Return GraphHit objects with a relevance score
        """
        keywords = _tokenise_query(query)
        if not keywords:
            return []

        seen: Dict[str, float] = {}   # node_id → best_score

        # Score each entity hit
        for kw in keywords:
            for ent in self.kg.search_entities(kw, top_k=top_k_entities):
                nid   = ent["node_id"]
                score = 1.0   # direct entity match

                if nid not in seen or seen[nid] < score:
                    seen[nid] = score

        # Also credit 1-hop neighbours
        for seed_id in list(seen.keys()):
            for edge in self.kg.get_neighbours(seed_id, depth=1):
                neighbour_id = edge["to"] if edge["from"] == seed_id else edge["from"]
                if neighbour_id not in seen:
                    seen[neighbour_id] = 0.7

        # Build GraphHit objects for top-k seeds
        top_seeds = sorted(seen.items(), key=lambda x: -x[1])[:top_k_entities]

        hits: List[GraphHit] = []
        for nid, score in top_seeds:
            ctx = self.kg.get_entity_context(nid)
            if not ctx:
                continue
            node_data = ctx.get("node", {})
            relations = ctx.get("outgoing", []) + ctx.get("incoming", [])

            hits.append(GraphHit(
                node_id     = nid,
                name        = node_data.get("name",        nid),
                entity_type = node_data.get("type",        "OTHER"),
                description = node_data.get("description", ""),
                relations   = relations,
                graph_score = score,
            ))

        return hits


# ---------------------------------------------------------------------------
# Fusion Engine
# ---------------------------------------------------------------------------

class FusionEngine:
    """
    Merges vector hits and graph hits into a single ranked list.

    Scoring:
        final_score = ALPHA * vector_score + BETA * graph_score

    Deduplication: chunks sharing the same chunk_id are merged.
    Graph signals are attached to the corresponding chunk result if the
    chunk mentions one of the retrieved entities.
    """

    # ------------------------------------------------------------------
    @staticmethod
    def fuse(vector_hits: List[VectorHit],
             graph_hits:  List[GraphHit],
             top_n: int = FUSION_TOP_N) -> List[FusedResult]:
        """
        Merge and rank vector + graph results.

        Strategy:
          - Each VectorHit becomes a FusedResult with vector_score set.
          - For each FusedResult, check if any GraphHit entity appears in
            the chunk text → add graph_score bonus.
          - Pure graph hits (entities in KG but not retrieved by vector)
            are added as graph-only results (vector_score=0).
        """
        # --- Anchor results on vector hits ---
        results: Dict[str, FusedResult] = {}

        for vh in vector_hits:
            cid = vh.chunk.chunk_id
            results[cid] = FusedResult(
                chunk        = vh.chunk,
                vector_score = vh.vector_score,
                graph_score  = 0.0,
                final_score  = ALPHA * vh.vector_score,
                graph_hits   = [],
            )

        # --- Attach graph signals to matching results ---
        for gh in graph_hits:
            entity_lower = gh.name.lower()

            # Find any already-retrieved chunk that mentions this entity
            matched = False
            for cid, result in results.items():
                if entity_lower in result.chunk.text.lower():
                    # Boost graph_score (take max if already set)
                    if result.graph_score < gh.graph_score:
                        result.graph_score = gh.graph_score
                    result.graph_hits.append(gh)
                    matched = True

            # If no vector result mentions this entity, create a graph-only entry
            if not matched:
                # Use entity description as a synthetic "chunk" (node_id as key)
                synthetic_key = f"__graph__{gh.node_id}"
                if synthetic_key not in results:
                    results[synthetic_key] = FusedResult(
                        chunk        = None,
                        vector_score = 0.0,
                        graph_score  = gh.graph_score,
                        final_score  = 0.0,
                        graph_hits   = [gh],
                    )

        # --- Recompute final scores ---
        for result in results.values():
            result.final_score = (
                ALPHA * result.vector_score + BETA * result.graph_score
            )

        # --- Sort by final score, return top N ---
        ranked = sorted(results.values(), key=lambda r: -r.final_score)
        return ranked[:top_n]


# ---------------------------------------------------------------------------
# Context Builder
# ---------------------------------------------------------------------------

def _build_graph_context(graph_hits: List[GraphHit]) -> str:
    """Format graph hits as a structured text block for prompt injection."""
    if not graph_hits:
        return "No graph context retrieved."

    lines = ["=== GRAPH KNOWLEDGE ==="]
    seen_entities = set()

    for gh in graph_hits:
        if gh.node_id in seen_entities:
            continue
        seen_entities.add(gh.node_id)

        lines.append(f"\n[{gh.entity_type}] {gh.name}")
        if gh.description:
            lines.append(f"  Description: {gh.description}")

        for rel in gh.relations[:4]:   # limit to 4 relations per entity
            direction = rel.get("direction", "out")
            neighbour = rel.get("neighbour", "?")
            relation  = rel.get("relation",  "?")
            context   = rel.get("context",   "")

            if direction == "out":
                lines.append(f"  → {relation} → {neighbour}")
            else:
                lines.append(f"  ← {relation} ← {neighbour}")

            if context:
                ctx_short = (context[:120] + "...") if len(context) > 120 else context
                lines.append(f'     ("{ctx_short}")')

    return "\n".join(lines)


def _build_vector_context(fused_results: List[FusedResult]) -> str:
    """Format top text chunks as a structured block for prompt injection."""
    lines = ["=== RETRIEVED TEXT CHUNKS ==="]

    for i, result in enumerate(fused_results, 1):
        if result.chunk is None:
            continue
        chunk = result.chunk
        lines.append(
            f"\n[Chunk {i}] {chunk.paper_name} | {chunk.section} "
            f"(score={result.final_score:.3f})"
        )
        lines.append(chunk.text[:800] + ("..." if len(chunk.text) > 800 else ""))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API: HybridRetriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Main entry point for the Retrieval Layer.

    Usage:
        retriever = HybridRetriever(vector_index, knowledge_graph)
        result    = retriever.retrieve("How does BERT compare to GPT?")
    """

    def __init__(self,
                 vector_index:     VectorIndex,
                 knowledge_graph:  KnowledgeGraph):
        self.vi = vector_index
        self.gr = GraphRetriever(knowledge_graph)
        self.fe = FusionEngine()

    # ------------------------------------------------------------------
    def retrieve(self, query: str,
                 top_k_vector:  int = VECTOR_TOP_K,
                 top_k_graph:   int = GRAPH_TOP_K,
                 top_n_results: int = FUSION_TOP_N) -> RetrievalResult:
        """
        Execute hybrid retrieval for `query`.

        Steps:
          1. Vector search → top-k text chunks
          2. Graph search  → top-k entity seeds + neighbourhood
          3. Fuse results  → ranked FusedResult list
          4. Build context strings
          5. Return RetrievalResult
        """
        # Step 1: Vector
        vector_hits = self.vi.search(query, top_k=top_k_vector)

        # Step 2: Graph
        graph_hits  = self.gr.search(query, top_k_entities=top_k_graph)

        # Step 3: Fusion
        fused = self.fe.fuse(vector_hits, graph_hits, top_n=top_n_results)

        # Step 4: Context
        all_graph_hits = [gh for fr in fused for gh in fr.graph_hits]
        # Also add graph hits that aren't in fused (graph-only, below threshold)
        present_ids = {gh.node_id for gh in all_graph_hits}
        for gh in graph_hits:
            if gh.node_id not in present_ids:
                all_graph_hits.append(gh)

        graph_context  = _build_graph_context(all_graph_hits)
        vector_context = _build_vector_context(fused)

        full_context = (
            f"USER QUERY: {query}\n\n"
            f"{graph_context}\n\n"
            f"{vector_context}"
        )

        return RetrievalResult(
            query          = query,
            fused          = fused,
            graph_context  = graph_context,
            vector_context = vector_context,
            full_context   = full_context,
        )