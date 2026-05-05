import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Mixing V1 models and V2 models")

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool

# Load environment variables
load_dotenv()

#Custom packages 
from older_version.agent_stories import websearch_backstory
from utils import arxiv_search , get_llm_client, execute_plotting_code
from older_version.agent_params import web_search, hypothesis_agent, visualization_agent , graphrag_builder, research_analyst_graph


# -------------------------------------------------
# Crew creation (CrewAI 0.86.0 with proper tools)
# -------------------------------------------------
def create_research_crew(query: str):
    llm_client = get_llm_client()

    # -----------------------------
    # Agents (CrewAI 0.86.0 with decorated tools)
    # -----------------------------
    web_searcher = web_search

    research_analyst = Agent(
        role="Research Analyst",
        goal="Analyze and synthesize information into accurate, structured insights with citations.",
        backstory="Expert at fact-checking, analysis, and insight extraction.",
        llm=llm_client,
        verbose=True,
        allow_delegation=True,
    )

    coding_agent = Agent(
        role="Coding Agent",
        goal="Write clean, correct, and efficient code based on the verified research analysis.",
        backstory=(
            "Senior software engineer who converts requirements and analysis "
            "into production-ready code and examples."
        ),
        llm=llm_client,
        verbose=True,
        allow_delegation=False,
    )

    technical_writer = Agent(
        role="Technical Writer",
        goal="Create a clear, well-structured markdown response with explanations, code, and citations.",
        backstory="Expert at explaining complex technical topics clearly and concisely.",
        llm=llm_client,
        verbose=True,
        allow_delegation=False,
    )

    # -----------------------------
    # Tasks (CrewAI 0.86.0)
    # -----------------------------
    hypothesis_task = Task(
        description=(
            f"Analyze the following research query: '{query}'. "
            "Formulate a highly technical, and testable hypotheses "
            "or architectural approaches to solve this problem. Detail the theoretical "
            "justification for each."
        ),
        agent=hypothesis_agent,
        expected_output="A structured , detailed hypotheses with theoretical justifications."
    )

    search_task = Task(
        description=f"Search for comprehensive and up-to-date information about: {query}. Use the arxiv_search tool to perform web searches and gather relevant research paper information with source links.",
        agent=web_searcher,
        expected_output="Raw search results with source links and references.",
        tools=[arxiv_search],
        context=[hypothesis_task],
    )

    graphrag_task = Task(
        description=(
            "Take the JSON paper list and build a GraphRAG knowledge graph. "
            "Store all nodes (papers, concepts, authors) and edges in Neo4j. "
            "Return the graph summary (node counts, concept list)."
        ),
        expected_output="Confirmation that graph was stored + a summary of nodes and concepts.",
        agent=graphrag_builder,
        context=[search_task],

    )

    analysis_task = Task(
        description=(
            "Analyze the search results, verify facts, resolve inconsistencies, "
            "and produce structured insights with citations."
        ),
        agent=research_analyst,
        expected_output="Structured analysis with verified insights and sources.",
        context=[search_task],
    )
    graph_analysis_task = Task(
        description=(
            "Use the Graph Query Tool to explore the Neo4j knowledge graph for: {topic}. "
            "Query key concepts from the graph summary. Identify dominant methods, "
            "author clusters, conflicts, and gaps. Cite every claim with a paper title."
        ),
        expected_output="Structured research report with Overview, Key Concepts, Conflicts, Open Questions.",
        agent=research_analyst,
        context=[graphrag_task],    # gives analyst the concept list to query
    )
    coding_task = Task(
        description=(
            "Based on the research analysis, write any necessary code, examples, "
            "or technical implementations needed to address the query."
        ),
        agent=coding_agent,
        expected_output="Executable, well-documented code snippets and examples.",
        context=[analysis_task],
    )
    visualization_task = Task(
        description=(
            "Review the verified insights from the Research Analyst. Identify opportunities "
            "to visually represent key data points, variances, or structural arguments "
            "made in the analyzed research papers. "
            
            "Based on that analysis, create 1 or 2 distinct Python visualization examples "
            "designed to highlight these concepts. Create the necessary synthetic data representation "
            "inside the code that mimics the analyst's findings."
            
            "**Use the 'Python Plotting Executor' tool** to generate and save these actual graph files. "
            "Do NOT include plt.show(). The output must be the result of the tool execution."
        ),
        agent=visualization_agent,
        expected_output="Filepaths of successfully generated PNG plot files with brief descriptions of what they visualize.",
        context=[analysis_task], # Needs the analyzed data
    )

    # writing_task = Task(
    #     description=(
    #         "Create a comprehensive, well-organized markdown response that combines "
    #         "research insights, explanations, and code with proper citations."
    #     ),
    #     agent=technical_writer,
    #     expected_output="Final markdown response with explanations, code, and citations.",
    #     context=[analysis_task, coding_task],
    # )

    writing_task = Task(
        description=(
            "Create a comprehensive, well-organized markdown response that combines "
            "research insights, explanations, technical implementations (code snippets), "
            "AND embed any generated visualizations."
        ),
        agent=technical_writer,
        expected_output="Final markdown response. Embed generated graphs using Markdown syntax: ![Plot Description](path/to/image.png)",
        # ✅ Add visualization_task to context!
        context=[analysis_task, coding_task, visualization_task], 
    )
    # -----------------------------
    # Crew (CrewAI 0.86.0)
    # -----------------------------
    return Crew(
        agents=[
            hypothesis_agent,
            web_searcher,
            research_analyst,
            coding_agent,
            visualization_agent,
            technical_writer,
        ],

        tasks=[
            hypothesis_task,
            search_task,
            graph_analysis_task,
            coding_task,
            visualization_task,
            writing_task,
        ],
        process=Process.sequential,
        verbose=True,
    )


# -------------------------------------------------
# Entry point 
# -------------------------------------------------
def run_research(query: str):
    """
    Run the research + coding workflow and return the final output.
    """
    try:
        crew = create_research_crew(query)
        result = crew.kickoff()
        return str(result) if result else "No results returned"
    except Exception as e:
        return f"Error: {str(e)}"