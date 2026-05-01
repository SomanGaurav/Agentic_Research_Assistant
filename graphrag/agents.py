"""
=============================================================================
LAYER 4: AGENT LAYER
=============================================================================
Responsibilities:
  - Provide specialised agents that consume RetrievalResult objects
  - Orchestrate agent selection based on query intent classification
  - Keep agents stateless (each call is independent)

Agents:
  1. SummarizerAgent      → concise, structured summary of retrieved content
  2. TechnicalWriterAgent → detailed analysis, comparisons, literature review
  3. VisualizerAgent      → structured JSON graph/diagram + Mermaid notation

Orchestration:
  - QueryRouter classifies intent via Claude
  - AgentOrchestrator selects and invokes the right agent(s)
  - Results from multiple agents can be chained (e.g. summarise + visualise)

Design:
  - All agents share a BaseAgent interface: agent.run(query, context) → AgentResponse
  - No agent directly accesses Ingestion or Graph layers
  - Router is a lightweight Claude call (no retrieval needed)
=============================================================================
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import anthropic

from retriever import RetrievalResult


# ---------------------------------------------------------------------------
# Shared Data Models
# ---------------------------------------------------------------------------

class AgentType(str, Enum):
    SUMMARIZER      = "summarizer"
    TECHNICAL_WRITER = "technical_writer"
    VISUALIZER      = "visualizer"


@dataclass
class AgentResponse:
    agent_type: AgentType
    query:      str
    output:     str          # main text/markdown/JSON output
    metadata:   Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        separator = "=" * 60
        return (
            f"\n{separator}\n"
            f"Agent: {self.agent_type.value.upper()}\n"
            f"Query: {self.query}\n"
            f"{separator}\n"
            f"{self.output}\n"
        )


# ---------------------------------------------------------------------------
# Base Agent
# ---------------------------------------------------------------------------

class BaseAgent(ABC):
    """
    All agents must implement `run()`.
    They receive only the query string and a RetrievalResult — nothing else.
    """

    def __init__(self, client: Optional[anthropic.Anthropic] = None,
                 model: str = "claude-sonnet-4-20250514"):
        self.client = client or anthropic.Anthropic()
        self.model  = model

    # ------------------------------------------------------------------
    @abstractmethod
    def agent_type(self) -> AgentType:
        ...

    @abstractmethod
    def system_prompt(self) -> str:
        ...

    # ------------------------------------------------------------------
    def run(self, query: str, context: RetrievalResult) -> AgentResponse:
        """Execute the agent. Subclasses may override for custom logic."""
        user_prompt = self._build_prompt(query, context)

        response = self.client.messages.create(
            model      = self.model,
            max_tokens = 2000,
            system     = self.system_prompt(),
            messages   = [{"role": "user", "content": user_prompt}],
        )

        output = response.content[0].text.strip()

        return AgentResponse(
            agent_type = self.agent_type(),
            query      = query,
            output     = output,
            metadata   = {
                "input_tokens":  response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )

    # ------------------------------------------------------------------
    def _build_prompt(self, query: str, context: RetrievalResult) -> str:
        return (
            f"USER QUERY:\n{query}\n\n"
            f"RETRIEVED CONTEXT:\n{context.full_context}"
        )


# ---------------------------------------------------------------------------
# Agent 1: Summarizer
# ---------------------------------------------------------------------------

class SummarizerAgent(BaseAgent):
    """
    Produces a concise, well-structured summary of the retrieved context.
    Output format: 3–5 bullet key findings + 1 paragraph synthesis.
    Target audience: researcher who wants a quick overview.
    """

    def agent_type(self) -> AgentType:
        return AgentType.SUMMARIZER

    def system_prompt(self) -> str:
        return """You are a scientific summarization expert.

Given a user query and retrieved context from research papers (including both
text chunks and a knowledge graph), produce a concise, accurate summary.

Your response MUST follow this exact structure:

## Key Findings
- [Bullet 1: most important finding, ≤ 2 sentences]
- [Bullet 2]
- [Bullet 3]
- [Bullet 4, if applicable]
- [Bullet 5, if applicable]

## Summary
[One focused paragraph (5–8 sentences) synthesising the findings above,
directly answering the user's query. Cite paper names when possible.]

## Source Papers
[Comma-separated list of paper names referenced in the context]

Rules:
- Be specific: use exact model names, metric values, dataset names from context
- Do NOT hallucinate information not present in the context
- Be concise — summary paragraph must be ≤ 150 words
- If context is insufficient, state what is missing
"""


# ---------------------------------------------------------------------------
# Agent 2: Technical Writer
# ---------------------------------------------------------------------------

class TechnicalWriterAgent(BaseAgent):
    """
    Produces detailed technical explanations, comparisons, or literature
    review sections.
    Output format: structured markdown with sections and technical depth.
    Target audience: researcher writing a paper or a detailed report.
    """

    def agent_type(self) -> AgentType:
        return AgentType.TECHNICAL_WRITER

    def system_prompt(self) -> str:
        return """You are an expert technical writer specialising in AI/ML research.

Given a user query and retrieved context from research papers (text chunks +
knowledge graph), produce a detailed, technically rigorous response.

Format your response as structured markdown:

## Overview
[2–3 sentences establishing the topic and scope]

## Technical Analysis
[Detailed explanation, 3–5 paragraphs. Include:
 - how methods/models work
 - quantitative results and comparisons if available
 - technical trade-offs
 - key design decisions]

## Comparative Analysis (if multiple approaches present)
| Approach | Key Feature | Dataset | Metric | Score |
|----------|-------------|---------|--------|-------|
[Fill table from context. Skip if only one approach present.]

## Key Relationships (from Knowledge Graph)
[2–3 sentences describing the most important entity relationships found
in the graph context, e.g. "Model X uses Dataset Y and outperforms Z"]

## Gaps and Open Questions
[1–2 sentences on what the papers do not address or leave open]

Rules:
- Use technical vocabulary appropriate for an ML/NLP audience
- Cite paper names explicitly (e.g., "Smith et al. (2023) show…")
- Preserve exact metric names and values from the context
- Do NOT fabricate results not present in the retrieved context
"""


# ---------------------------------------------------------------------------
# Agent 3: Visualizer
# ---------------------------------------------------------------------------

_VISUALIZER_SCHEMA = """
{
  "title": "string — short title for the diagram",
  "diagram_type": "one of: entity_relation | comparison_table | concept_hierarchy | timeline",
  "mermaid_code": "string — valid Mermaid diagram code",
  "structured_data": {
    "nodes": [{"id": "string", "label": "string", "type": "string"}],
    "edges": [{"from": "string", "to": "string", "label": "string"}]
  },
  "description": "string — 2-3 sentences explaining what the diagram shows"
}
"""

class VisualizerAgent(BaseAgent):
    """
    Produces structured graph/diagram representations of the retrieved
    knowledge.
    Output: JSON with Mermaid diagram code + structured node/edge data.
    Target audience: researcher wanting a visual overview of relationships.
    """

    def agent_type(self) -> AgentType:
        return AgentType.VISUALIZER

    def system_prompt(self) -> str:
        return (
            "You are a scientific knowledge visualisation expert.\n\n"
            "Given a user query and retrieved context from research papers "
            "(text + knowledge graph), produce a structured visual representation.\n\n"
            "Return ONLY valid JSON matching this schema — no markdown fences:\n"
            + _VISUALIZER_SCHEMA
            + "\n\nMermaid diagram rules:\n"
            "- For entity_relation: use 'graph LR' with labelled arrows\n"
            "- For comparison_table: use a Mermaid table or 'flowchart TD' with boxes\n"
            "- For concept_hierarchy: use 'graph TD' tree structure\n"
            "- Keep diagrams readable: ≤ 15 nodes\n"
            "- Node IDs must be alphanumeric (no spaces)\n"
            "- Use the entities and relationships from the graph context\n"
            "- Do NOT hallucinate entities not present in the context\n"
        )

    def run(self, query: str, context: RetrievalResult) -> AgentResponse:
        """Override to parse JSON output and pretty-print."""
        response = super().run(query, context)

        raw = response.output

        # Strip any accidental markdown fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw)

        try:
            parsed = json.loads(raw)
            formatted = json.dumps(parsed, indent=2)

            mermaid = parsed.get("mermaid_code", "")
            description = parsed.get("description", "")

            output = (
                f"**Diagram: {parsed.get('title', 'Knowledge Graph')}**\n\n"
                f"{description}\n\n"
                f"```mermaid\n{mermaid}\n```\n\n"
                f"<details><summary>Structured Data (JSON)</summary>\n\n"
                f"```json\n{formatted}\n```\n</details>"
            )

        except (json.JSONDecodeError, KeyError):
            output = f"[Visualizer — raw output]\n{raw}"

        response.output = output
        return response


# ---------------------------------------------------------------------------
# Query Router
# ---------------------------------------------------------------------------

_ROUTER_SYSTEM = """You are a query intent classifier for a research paper analysis system.

Classify the user's query into one or more agent types:
  - "summarizer"       : user wants a concise summary or overview
  - "technical_writer" : user wants detailed explanation, comparison, literature review, or analysis
  - "visualizer"       : user wants a diagram, graph, visual, or structured relationship view

Return ONLY valid JSON — no markdown, no preamble:
{"agents": ["agent_type_1", "agent_type_2"]}

Rules:
- Return exactly the agents needed (1–3)
- Queries asking for "both summary and diagram" → ["summarizer", "visualizer"]
- Queries like "compare", "analyse", "explain how" → ["technical_writer"]
- Queries like "summarise", "overview", "what is" → ["summarizer"]
- Queries like "show relationships", "draw", "graph" → ["visualizer"]
- When unsure, prefer ["summarizer"]
"""

class QueryRouter:
    """
    Lightweight classifier that decides which agent(s) to invoke.
    Uses Claude with a minimal prompt — no retrieval context needed.
    """

    def __init__(self, client: Optional[anthropic.Anthropic] = None):
        self.client = client or anthropic.Anthropic()

    def route(self, query: str) -> List[AgentType]:
        """Return a list of AgentType values to invoke for this query."""
        response = self.client.messages.create(
            model      = "claude-sonnet-4-20250514",
            max_tokens = 100,
            system     = _ROUTER_SYSTEM,
            messages   = [{"role": "user", "content": f"Query: {query}"}],
        )
        raw = response.content[0].text.strip()

        try:
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            parsed = json.loads(raw)
            agents = parsed.get("agents", ["summarizer"])
        except (json.JSONDecodeError, KeyError):
            agents = ["summarizer"]

        type_map = {
            "summarizer":       AgentType.SUMMARIZER,
            "technical_writer": AgentType.TECHNICAL_WRITER,
            "visualizer":       AgentType.VISUALIZER,
        }
        return [type_map[a] for a in agents if a in type_map]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class AgentOrchestrator:
    """
    Top-level controller for the Agent Layer.

    Usage:
        orchestrator = AgentOrchestrator()
        responses    = orchestrator.run(query, retrieval_result)

    The orchestrator:
      1. Routes the query to the appropriate agent(s)
      2. Invokes each agent sequentially
      3. Returns a list of AgentResponse objects
    """

    def __init__(self, client: Optional[anthropic.Anthropic] = None):
        self.client = client or anthropic.Anthropic()
        self.router = QueryRouter(client=self.client)

        self._agents: Dict[AgentType, BaseAgent] = {
            AgentType.SUMMARIZER:       SummarizerAgent(client=self.client),
            AgentType.TECHNICAL_WRITER: TechnicalWriterAgent(client=self.client),
            AgentType.VISUALIZER:       VisualizerAgent(client=self.client),
        }

    # ------------------------------------------------------------------
    def run(self, query: str,
            context: RetrievalResult,
            force_agents: Optional[List[AgentType]] = None) -> List[AgentResponse]:
        """
        Execute the full agent pipeline.

        Args:
            query        : user's natural-language question
            context      : RetrievalResult from the HybridRetriever
            force_agents : if set, skip routing and use these agents directly

        Returns:
            list of AgentResponse objects (one per invoked agent)
        """
        if force_agents:
            selected = force_agents
            print(f"  [Orchestrator] Forced agents: {[a.value for a in selected]}")
        else:
            selected = self.router.route(query)
            print(f"  [Orchestrator] Routed to: {[a.value for a in selected]}")

        responses: List[AgentResponse] = []
        for agent_type in selected:
            agent = self._agents[agent_type]
            print(f"  [Orchestrator] Running {agent_type.value}…")
            response = agent.run(query, context)
            responses.append(response)

        return responses

    # ------------------------------------------------------------------
    def run_all(self, query: str,
                context: RetrievalResult) -> List[AgentResponse]:
        """Convenience: run all three agents regardless of routing."""
        return self.run(query, context,
                        force_agents=list(AgentType))