from crewai import Agent 
from crewai.tools import tool
from dotenv import load_dotenv
from utils import arxiv_search , get_llm_client, execute_plotting_code , GraphRAGTool , GraphQueryTool
from older_version.agent_stories import websearch_backstory, hypothesis_backstory, visualization_backstory , graphrag_backstory

llm_client = get_llm_client()

web_search = Agent(
        role="Research Paper Finder",
        goal="Find highly relevant research papers from arXiv with titles, authors, summaries, and links.",
        backstory=websearch_backstory,
        tools=[arxiv_search],
        llm=llm_client,
        verbose=True,
        allow_delegation=True,
    )

hypothesis_agent = Agent(
    role="Lead Hypothesis Strategist",
    goal="Formulate 1 distinct, testable hypotheses or technical approaches based on the user's query.",
    backstory=hypothesis_backstory,
    llm=llm_client,
    verbose=True,
    allow_delegation=False, # It shouldn't pass the buck; it needs to think.
)

visualization_agent = Agent(
        role="Data Visualization Specialist",
        goal="Analyze research data/insights and use Python to generate impactful scientific graphs (e.g., charts, matrices, distributions) as files.",
        backstory=visualization_backstory,
        tools=[execute_plotting_code], # <--- Crucial Tool!
        llm=llm_client,
        verbose=True,
        allow_delegation=False,
    )




graphrag_tool = GraphRAGTool() 


graphrag_builder = Agent(
    role="Knowledge Graph Engineer",
    goal=(
        "Transform raw research papers into a structured GraphRAG knowledge graph "
        "by extracting entities (concepts, authors, methods) and their relationships "
        "(cites, extends, contradicts, uses). Output a graph context that the Research "
        "Analyst can use for deep, citation-aware synthesis."
    ),
    backstory=graphrag_backstory,
    tools=[graphrag_tool],
    llm=llm_client,
    verbose=True,
    allow_delegation=False,   # owns its output; no need to delegate
)

graphrag_builder = Agent(
    role="Knowledge Graph Engineer",
    goal="Build and persist a GraphRAG knowledge graph to Neo4j from research papers.",
    backstory="Expert in graph-based knowledge representation and Neo4j.",
    tools=[graphrag_tool],          # writes to Neo4j
    llm=llm_client,
    verbose=True,
    allow_delegation=False,
)


graph_query_tool = GraphQueryTool()
research_analyst_graph = Agent(
    role="Research Analyst",
    goal="Query the Neo4j knowledge graph to synthesize citation-backed research insights.",
    backstory="Expert at fact-checking, analysis, and insight extraction from graph data.",
    tools=[graph_query_tool],       # ← reads from Neo4j
    llm=llm_client,
    verbose=True,
    allow_delegation=True,
)