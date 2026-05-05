# from crewai import Agent
# from utils import get_llm_client 

# manager_agent_backstory = (
#     "You are a seasoned operations director specializing in research systems and intelligent task routing. "
#     "With a background in information science and decision systems, you excel at quickly identifying whether "
#     "a request requires discovery or synthesis. You never perform the research yourself—instead, you act as "
#     "the central coordinator who ensures every query is handled by the most suitable specialist. "
#     "When a user provides a broad topic or open-ended question, you immediately engage the Search Agent "
#     "to gather relevant sources. When the user provides a specific paper, dataset, or text, you bypass search "
#     "entirely and assign the task to the Researcher Agent for deep analysis and summarization. "
#     "Your priority is speed, clarity, and optimal delegation, ensuring a streamlined and intelligent workflow "
#     "every time."
# )

# manager_agent = Agent(
#     role="Research Operations Director",
#     goal=(
#         "Classify user intent and delegate tasks STRICTLY as follows:\n\n"
#         "1. If the user provides a general topic → Delegate the task sequentially  first only to hypothesis_agent , then only to search_agent and atlast only to analysis_agent\n"
#         "2. If the user asks to only download a paper → Delegate task only to search_agent\n"
#         "3. If the user provides a specific document, text, or asks for summary → delegate task only to analysis_agent\n\n"
#         "Rules:\n"
#         "- NEVER perform the task yourself\n"
#         "- ALWAYS delegate to exactly one agent\n"
#         "- DO NOT mix responsibilities\n"
#     ),
#     backstory=(
#         "You are a strict dispatcher. You do not do any research. "
#         "You only assign work based on clear rules."
#     ),
#     allow_delegation=True,
#     llm=get_llm_client(),
# )
# agents/manager_agent.py
from crewai import Agent
from utils import get_llm_client


def make_manager_agent(available_agents: list) -> Agent:
    """
    Called by executor.py after the task list is built.
    Receives only the agents actually participating in this run,
    so the manager never tries to delegate to a non-existent agent.
    """

    # Build a readable roster from the live agent list
    agent_roster = "\n".join([
        f"- {agent.role}: {agent.goal}"
        for agent in available_agents
    ])

    return Agent(
        role="Research Operations Director",
        goal=(
            "Oversee task execution. Delegate every task to the correct agent "
            "from the roster below. Review outputs before marking tasks complete.\n\n"
            "Rules:\n"
            "- NEVER perform tasks yourself\n"
            "- ONLY delegate to agents listed in your roster\n"
            "- Validate each output before proceeding to the next task\n"
        ),
        backstory=(
            "You are a strict research dispatcher. You assign work and validate results.\n\n"
            f"Agents available to you right now:\n{agent_roster}\n\n"
            "You must only delegate to agents on this list."
        ),
        allow_delegation=True,
        llm=get_llm_client(),
    )