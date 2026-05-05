from crewai import Agent , Task 
from utils import get_llm_client 
from pydantic import BaseModel
from typing import List

llm_client = get_llm_client()

class Theme(BaseModel):
    name: str
    papers: List[str]
    description: str

class AnalysisOutput(BaseModel):
    themes: List[Theme]
    trends: List[str]
    gaps: List[str]


analyst_agent = Agent(
    role="research_analyst",
    goal="Synthesize multiple research papers into insights, trends, and gaps",
    backstory=(
        "You are an expert researcher who compares papers, identifies patterns, "
        "and extracts deep insights across multiple works."
    ),
    verbose=True , 
    llm=llm_client 
)

