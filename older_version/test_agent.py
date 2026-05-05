from crewai import Task, Crew
from dotenv import load_dotenv
load_dotenv()
from older_version.agent_params import web_search, hypothesis_agent, visualization_agent , graphrag_builder
from older_version.agent_stories import websearch_backstory
from utils import arxiv_search , get_llm_client
# Import all your agents here
# from agents.other_agent import other_agent
# from agents.xyz_agent import xyz_agent

# 🔹 Register agents in a dictionary
AGENTS = {
    "HYPOTHESIS": hypothesis_agent,
    "WEBSEARCH": web_search,
    "VISUALIZER": visualization_agent,
    "GRAPHRAG": graphrag_builder,
}


def test_agent(agent_name: str, query: str):
    agent = AGENTS.get(agent_name)

    if not agent:
        raise ValueError(f"Agent '{agent_name}' not found. Available: {list(AGENTS.keys())}")

    task = Task(
        description=query,
        agent=agent,
        expected_output="Structured response based on agent role"
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True
    )

    result = crew.kickoff()
    print("\n===== RESULT =====\n")
    print(result)


# 🔹 CLI-style usage
if __name__ == "__main__":
    print("Available agents:", list(AGENTS.keys()))
    
    agent_name = input("Enter agent name: ").strip()
    query = input("Enter task/query: ").strip()

    test_agent(agent_name, query)