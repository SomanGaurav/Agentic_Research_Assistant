import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Mixing V1 models and V2 models")

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool

# Load environment variables
load_dotenv()

#Custom packages 
from agent_stories import websearch_backstory
from utils import arxiv_search , get_llm_client, execute_plotting_code
from agent_params import web_searcher, hypothesis_agent, visualization_agent

# -------------------------------------------------
# LLM configuration (CrewAI 0.86.0)
# -------------------------------------------------
# def get_llm_client():
#     from crewai import LLM
#     return LLM(
#         model="ollama/tom_himanen/deepseek-r1-roo-cline-tools:1.5b",
#         base_url="http://localhost:11434"
#     )

# -------------------------------------------------
# Tool: LinkUp Search (using @tool decorator)
# -------------------------------------------------
@tool("LinkUp Web Search")
def linkup_search(query: str) -> str:
    """
    Search the web using LinkUp API and return comprehensive search results with sources.
    
    Args:
        query: The search query to look up information for
        
    Returns:
        Formatted search results with source links and relevant information
    """
    try:
        from linkup import LinkupClient
        client = LinkupClient(api_key=os.getenv("LINKUP_API_KEY"))
        response = client.search(
            query=query,
            depth="standard",
            output_type="searchResults"
        )
        return str(response)
    except Exception as e:
        return f"Error occurred while searching: {str(e)}"





# -------------------------------------------------
# Crew creation (CrewAI 0.86.0 with proper tools)
# -------------------------------------------------
def create_research_crew(query: str):
    llm_client = get_llm_client()

    # -----------------------------
    # Agents (CrewAI 0.86.0 with decorated tools)
    # -----------------------------
    web_searcher = web_searcher

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
            "Formulate 3 distinct, highly technical, and testable hypotheses "
            "or architectural approaches to solve this problem. Detail the theoretical "
            "justification for each."
        ),
        agent=hypothesis_agent,
        expected_output="A structured list of 3 detailed hypotheses with theoretical justifications."
    )

    search_task = Task(
        description=f"Search for comprehensive and up-to-date information about: {query}. Use the linkup_search tool to perform web searches and gather relevant information with source links.",
        agent=web_searcher,
        expected_output="Raw search results with source links and references.",
        tools=[linkup_search],
        context=[hypothesis_task],
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
            analysis_task,
            coding_task,
            visualization_task,
            writing_task,
        ],
        process=Process.sequential,
        verbose=True,
    )


# -------------------------------------------------
# Entry point (CrewAI 0.86.0 compatible)
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