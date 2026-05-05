from crewai import Agent
from utils import get_llm_client , LocalFolderReader , LocalFileReader 


llm_client = get_llm_client()
research_analyst = Agent(
    role="Research Analyst",
    goal=(
        "Read and extract content from local research papers.\n"
        "MANDATORY TOOL ORDER:\n"
        "1. Call LocalFolderReader with NO arguments to list available files.\n"
        "2. Call LocalFileReader with only the filename, not a full path.\n"
        "Never invent or guess file paths."
        "location with LocalFolderReader."
    ),
    backstory=(
        "You are a precise research analyst who reads papers methodically. "
        "You always locate a file before reading it — never the other way around. "
        "You treat LocalFolderReader as a mandatory first step, not an optional one."
    ),
    tools=[LocalFolderReader(), LocalFileReader()],   # folder listed first intentionally
    llm=get_llm_client(),
    verbose=True,
)
