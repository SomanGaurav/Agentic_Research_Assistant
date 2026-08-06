# planner.py
import json

from crewai.project import llm
from utils import get_llm_client
from task_registry import TASK_DESCRIPTIONS

client = get_llm_client()

# PLANNER_PROMPT = f"""
# You are a research workflow planner. Given a user request, produce a JSON execution plan.

# Available tasks:
# {json.dumps(TASK_DESCRIPTIONS, indent=2)}

# Rules:
# - Only include tasks that are actually needed
# - Tasks run in order; later tasks can use outputs of earlier ones
# - Inputs must match what each task needs (query, file_name, paper_id, focus)
# - Be minimal — never add tasks the user did not ask for
# - If the user uploaded a file, read_local_file must come before summarize or analyze_and_synthesize
# - generate_hypotheses ALWAYS requires a 'query' input — use the user's topic as the value
# - download_paper ALWAYS requires a 'paper_id' input — use the ArXiv ID from search results or the user's provided identifier
# - If the user asks for a summary or analysis without uploading a file, you must first search for relevant papers, then download the most relevant one, and then read it before summarizing or analyzing.
# - find_citations ALWAYS requires a 'query' input — use the paper name the user mentioned
# - find_references ALWAYS requires a 'query' input — use the paper name the user mentioned
# - You MUST use task names EXACTLY as they appear in the available tasks list above
# - Do NOT invent task names or paraphrase them

# Output ONLY valid JSON, no markdown, no explanation:
# {{
#   "intent_summary": "brief description of what the user wants",
#   "tasks": [
#     {{
#       "task_id": 1,
#       "name": "task_name_from_registry",
#       "reason": "why this task is needed",
#       "inputs": {{
#         "key": "value or $task_N_output to reference a prior result"
#       }}
#     }}
#   ]
# }}
# """


PLANNER_PROMPT = f"""
You are a research workflow planner. Given a user request, produce a JSON execution plan.

Available tasks:
{json.dumps(TASK_DESCRIPTIONS, indent=2)}

TASK SELECTION RULES:
- Only include tasks that are actually needed — be minimal
- You MUST use task names EXACTLY as they appear in the available tasks list
- Do NOT invent task names, paraphrase them, or combine them
- Use literature_review instead of search_arxiv when user asks for:
  a survey, overview, literature review, state of the art, 
  foundational papers, seminal works, or recent advances on a topic
- Use search_arxiv only for a quick targeted paper search with a specific title or narrow query
- literature_review handles all time constraints automatically — 
  do NOT add separate tasks for time filtering

INPUT RULES:
- search_arxiv ALWAYS requires a 'query' input — use the user's topic
- generate_hypotheses ALWAYS requires a 'query' input — use the user's topic
- find_citations ALWAYS requires a 'query' input — use the exact paper name the user mentioned
- find_references ALWAYS requires a 'query' input — use the exact paper name the user mentioned
- download_paper ALWAYS requires a 'paper_id' input — use the ArXiv ID or paper title
- read_local_file ALWAYS requires a 'file_name' input — use the uploaded file name from context
- explain_concept ALWAYS requires a 'topic' input — use the EXACT concept the user asked about, not a broader topic

ORDERING RULES:
- read_local_file MUST come before explain_paper, visualize_results, or analyze_and_synthesize when a file is uploaded
- search_arxiv MUST come before analyze_and_synthesize, compare_papers, or download_paper
- find_citations and find_references do NOT need search_arxiv — they use Semantic Scholar directly

TASK BOUNDARY RULES — read carefully before choosing a task:
- Use explain_concept when user asks about a topic or concept with NO paper uploaded
- Use explain_paper when user has uploaded or downloaded a SPECIFIC paper
- Use analyze_and_synthesize ONLY across MULTIPLE papers from search results — never for a single paper
- Use compare_papers ONLY when user explicitly asks for a comparison between papers
- Do NOT call both explain_paper and analyze_and_synthesize for the same paper
- Do NOT call explain_concept and explain_paper together — pick one based on whether a paper exists

WHEN NO FILE IS UPLOADED:
- If user asks to summarize or analyze a topic without uploading a file:
  search_arxiv → download_paper → read_local_file → explain_paper

OUTPUT FORMAT:
Output ONLY valid JSON with no markdown, no explanation, no code fences:
{{
  "intent_summary": "one line description of what the user wants",
  "tasks": [
    {{
      "task_id": 1,
      "name": "exact_task_name",
      "reason": "one line reason why this task is needed",
      "inputs": {{
        "key": "value or $task_N_output to reference a prior result"
      }}
    }}
  ]
}}
"""

def plan(user_input: str, file_name: str = None) -> dict:
    # Give the planner awareness of any uploaded file
    context_note = ""
    if file_name:
        context_note = f"\n\nNote: The user has uploaded a file named '{file_name}'."

    response = client.call(                      # ← changed
        messages=[
            {"role": "system", "content": PLANNER_PROMPT},
            {"role": "user",   "content": user_input + context_note}
        ]
    )

    raw = response.strip() 
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Strip accidental markdown fences if model adds them
        clean = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)