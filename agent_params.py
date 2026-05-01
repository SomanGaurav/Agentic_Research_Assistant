from crewai import Agent 
from crewai.tools import tool
from dotenv import load_dotenv
from utils import arxiv_search , get_llm_client, execute_plotting_code
from agent_stories import websearch_backstory, hypothesis_backstory, visualization_backstory

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
    goal="Formulate 3 distinct, testable hypotheses or technical approaches based on the user's query.",
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

