# import os 
# import dotenv
# from crewai import Agent, Task, Crew, Process
# from agents.synthesis_agent import research_analyst
# from agents.search_agent import web_search
# from agents.hypothesis_agent import hypothesis_agent
# from agents.manager_agent import manager_agent
# from agents.analyst_agent import analyst_agent
# from utils import arxiv_search  



# dotenv.load_dotenv()

# ### Task Definition 



# manager_task = Task(
#     description=(
#         "1. Analyze the user's request: '{user_input}'.\n"
#         "2. Determine the workflow: \n"
#         "   - If the user provides a topic, delegate to the ArXiv Search Agent to find papers.\n"
#         "   - If the user provides a filename or specific content, delegate to the Researcher Agent for synthesis.\n"
#         "   - If both are needed, coordinate the sequence (Search then Synthesize).\n"
#         "3. Review the Researcher's output to ensure the Methodology and Results sections are accurate.\n"
#         "4. Provide the final, consolidated research report back to the user."
#     ),
#     agent=manager_agent,
#     expected_output="A final, high-quality academic report that directly addresses the user's query '{user_input}'."
# )



# hypothesis_task = Task(
#     description=(
#         "Analyze the following research query: '{query}'. "
#         "Formulate 3 distinct, highly technical, and testable hypotheses "
#         "or architectural approaches to solve this problem. Detail the theoretical justification."
#     ),
#     agent=hypothesis_agent,
#     expected_output="A structured list of 3 detailed hypotheses."
# )

# search_task = Task(
#     description="Search for comprehensive and up-to-date information about:{query}.",
#     agent=web_search,  
#     expected_output="Raw search results with source links and references.",
#     tools=[arxiv_search],
# )

# local_read_task = Task(
#     description=(
#         "1. Identify the file '{file_name}' using your LocalFolderReader tool.\n"
#         "2. Use the LocalFileReader tool to extract the Abstract, Methodology, and Results.\n"
#         "3. Following your 'Interpretive Extraction' and 'Verbatim Reporting' protocols, "
#         "prepare this content for final synthesis."
#     ),
#     expected_output="A structured extraction of {file_name} with verbatim results.",
#     agent=research_analyst  
# )


# analysis_task = Task(
#     description=(
#         "Analyze the provided research materials. "
#         "1. If papers were found via search, analyze those results. "
#         "2. If a local paper was read, analyze that content. "
#         "3. If both exist, synthesize them together. "
#         "Verify facts, resolve inconsistencies, and cite every claim with a paper title."
#     ),
#     agent=research_analyst,
#     expected_output="Structured analysis with verified insights and sources.",
#     context=[search_task, local_read_task] 
# )




# ### Defining the Crew 
# research_crew = Crew(
#     agents =[hypothesis_agent, web_search, research_analyst ],
#     tasks=[manager_task],
#     process=Process.hierarchical,
#     verbose=True,  # Set the manager agent for workflow coordination
#     manager_agent=manager_agent
# )
 
# main.py
from planner import plan
from executor import execute_plan


def run(user_input: str, file_name: str = None) -> str:

    # ── Step 1: LLM decides which tasks are needed ────────────────────────────
    execution_plan = plan(user_input, file_name=file_name)

    print("\nExecution Plan:")
    for step in execution_plan["tasks"]:
        print(f"   {step['task_id']}. {step['name']} → {step['reason']}")

    # ── Step 2: Build and run only those tasks ────────────────────────────────
    initial_context = {}
    if file_name:
        initial_context["uploaded_file"] = file_name

    result = execute_plan(execution_plan, initial_context)
    return result


# ── Examples ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Case 1: User uploaded a file, wants a summary
    # Planner produces → [read_local_file → summarize]
    print(run(
        user_input="just summarise this paper for me",
        file_name="attention_is_all_you_need.pdf"
    ))

    # Case 2: Search only, no download
    # Planner produces → [search_arxiv]
    print(run(
        user_input="find recent papers on LoRA fine-tuning"
    ))

    # Case 3: Full pipeline
    # Planner produces → [search_arxiv → download_paper → analyze_and_synthesize]
    print(run(
        user_input="find papers on diffusion models, download the top one and give me a deep analysis"
    ))

    # Case 4: Something you never hardcoded
    # Planner produces → [generate_hypotheses → search_arxiv → analyze_and_synthesize]
    print(run(
        user_input="generate hypotheses on MoE architectures then find supporting papers"
    ))