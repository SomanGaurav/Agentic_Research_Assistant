import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Mixing V1 models and V2 models")

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process

# Load environment variables
load_dotenv()

# Custom packages 
from utils import LocalFolderReader ,LocalFileReader , arxiv_search, get_llm_client, execute_plotting_code
from older_version.agent_params import (
    web_search, hypothesis_agent, visualization_agent
)

# -------------------------------------------------
# PHASE 1: RESEARCH CREW (Standard Analysis)
# -------------------------------------------------
def run_research_phase(query: str):
    llm_client = get_llm_client()

    # Reintroduced your standard Research Analyst
    research_analyst = Agent(
        role="Research Analyst",
        goal="Analyze and synthesize information into accurate, structured insights with citations.",
        backstory="Expert at fact-checking, analysis, and insight extraction.",
        llm=llm_client,
        verbose=True,
        allow_delegation=True,
    )

    hypothesis_task = Task(
        description=(
            f"Analyze the following research query: '{query}'. "
            "Formulate 3 distinct, highly technical, and testable hypotheses "
            "or architectural approaches to solve this problem. Detail the theoretical justification."
        ),
        agent=hypothesis_agent,
        expected_output="A structured list of 3 detailed hypotheses."
    )

    search_task = Task(
        description=f"Search for comprehensive and up-to-date information about: {query}.",
        agent=web_search,
        expected_output="Raw search results with source links and references.",
        tools=[arxiv_search],
        context=[hypothesis_task],
    )

    # Swapped GraphRAG tasks back to the standard Analysis task
    folder_tool = LocalFolderReader(folder_path="./papers")
    reader_tool = LocalFileReader(folder_path="./papers")
    analysis_task = Task(
        description=(
            "Analyze the search results, verify facts, resolve inconsistencies, "
            "and produce structured insights with citations. Identify dominant methods, "
            "conflicts, and gaps. Cite every claim with a paper title."
        ),
        agent=research_analyst,
        expected_output="Structured analysis with verified insights and sources.",
        tools=[folder_tool, reader_tool],  # <-- Standard analysis tools
        context=[search_task],
    )

    research_crew = Crew(
        agents=[hypothesis_agent, web_search, research_analyst],
        tasks=[hypothesis_task, search_task, analysis_task],
        process=Process.sequential,
        verbose=True,
        manager_agent=None,  # No manager for this standard analysis phase
    )
    
    result = research_crew.kickoff()
    return str(result) if result else "No research results returned."


# -------------------------------------------------
# PHASE 2: IMPLEMENTATION CREW
# -------------------------------------------------
def run_implementation_phase(research_context: str, user_directive: str):
    llm_client = get_llm_client()

    # Define the builder agents locally so they don't get tangled in Phase 1
    coding_agent = Agent(
        role="Coding Agent",
        goal="Write clean, correct, and efficient code based on verified research.",
        backstory="Senior software engineer who converts analysis into production-ready code.",
        llm=llm_client,
        verbose=True,
        allow_delegation=False,
    )

    technical_writer = Agent(
        role="Technical Writer",
        goal="Create a clear markdown response with explanations, code, and citations.",
        backstory="Expert at explaining complex technical topics clearly and concisely.",
        llm=llm_client,
        verbose=True,
        allow_delegation=False,
    )

    # Inject the research and the user's specific request into the tasks
    coding_task = Task(
        description=(
            f"Below is the finalized research context:\n{research_context}\n\n"
            f"The user has requested the following implementation: '{user_directive}'.\n"
            "Based on the research and this specific directive, write any necessary code, examples, "
            "or technical implementations."
        ),
        agent=coding_agent,
        expected_output="Executable, well-documented code snippets." ,
        context=[research_context]
    )

    visualization_task = Task(
        description=(
            f"Here is the finalized research context:\n{research_context}\n\n"
            "Review the research and the code generated. Create 1 or 2 distinct Python "
            "visualization examples designed to highlight the concepts the user asked for. "
            "**Use the 'Python Plotting Executor' tool** to generate and save these actual graph files.\n\n"
            "CRITICAL INSTRUCTION: When using the Python Plotting Executor tool, you MUST pass your input "
            "as a valid JSON dictionary with the key 'plotting_code'. Do not just output raw Python code. "
            "Ensure all newlines in the code are properly escaped. "
            "Do NOT include plt.show()."
        ),
        agent=visualization_agent,
        expected_output="Filepaths of successfully generated PNG plot files.",
        context=[coding_task], 
    )

    writing_task = Task(
        description=(
            "Create a comprehensive, well-organized markdown response that combines "
            "the implementation explanations, the code snippets, AND embed any generated visualizations."
        ),
        agent=technical_writer,
        expected_output="Final markdown response. Embed generated graphs using Markdown syntax: ![Plot Description](path/to/image.png)",
        context=[coding_task, visualization_task], 
    )

    implementation_crew = Crew(
        agents=[coding_agent, visualization_agent, technical_writer],
        tasks=[coding_task, visualization_task, writing_task],
        process=Process.sequential,
        verbose=True,
    )
    
    result = implementation_crew.kickoff()
    return str(result) if result else "No implementation results returned."