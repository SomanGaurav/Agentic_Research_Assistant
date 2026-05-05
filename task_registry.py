from crewai import Task
from agents.synthesis_agent import research_analyst
from agents.search_agent import web_search
from agents.hypothesis_agent import hypothesis_agent
from agents.analyst_agent import analyst_agent
from agents.visualization_agent import visualization_agent
from agents.analytics_agent import analytics_agent
from agents.explanation_agent import explainer_agent
from agents.concept_explainer_agent import concept_explainer_agent
from agents.citation_agent import citation_agent
from utils import arxiv_search



def make_search_task(query: str, **kwargs) -> Task:
    return Task(
        description=f"Search ArXiv for papers on: {query}",
        agent=web_search,
        tools=[arxiv_search],
        expected_output=(
            "List of relevant papers with titles, abstracts, and ArXiv IDs."
        )
    )

def make_download_task(paper_id: str, **kwargs) -> Task:
    return Task(
        description=f"Download the paper with ArXiv ID or title: {paper_id}",
        agent=research_analyst,
        expected_output="Confirmation of download and the saved local file path."
    )

def make_read_task(file_name: str, **kwargs) -> Task:
    return Task(
        description=(
            f"Read and extract content from the file '{file_name}'.\n\n"
            "YOU MUST FOLLOW THESE STEPS EXACTLY IN ORDER:\n"
            "STEP 1: Call LocalFolderReader first to locate '{file_name}' "
            "and confirm it exists in the papers directory. "
            "Wait for the result before proceeding.\n"
            "STEP 2: Only after STEP 1 succeeds, call LocalFileReader "
            "with the exact file path returned by LocalFolderReader.\n"
            "STEP 3: Extract the Abstract, Methodology, and Results verbatim "
            "from the file content returned in STEP 2.\n\n"
            "STRICT RULES:\n"
            "- Do NOT call LocalFileReader before LocalFolderReader\n"
            "- Do NOT guess or hardcode the file path\n"
            "- The file path passed to LocalFileReader must come "
            "from LocalFolderReader's output, not from the task input"
        ),
        agent=research_analyst,
        expected_output=f"Structured verbatim extraction from {file_name}."
    )

def make_summarize_task(focus: str = "general", **kwargs) -> Task:
    return Task(
        description=f"Summarize the research content. Focus on: {focus}.",
        agent=research_analyst,
        expected_output="A concise, well-structured summary of the research."
    )

def make_hypothesis_task(query: str, **kwargs) -> Task:
    return Task(
        description=(
            f"Formulate 3 distinct, testable hypotheses for: {query}. "
            "Include theoretical justification for each."
        ),
        agent=hypothesis_agent,
        expected_output="A structured list of 3 detailed hypotheses."
    )

def make_analysis_task(focus: str = "general", **kwargs) -> Task:
    return Task(
        description=(
            "Analyze and synthesize all research materials gathered so far. "
            f"Focus on: {focus}. Cite every claim with a paper title."
        ),
        agent=analyst_agent,
        expected_output="Structured analysis with verified insights and citations."
    )
def make_compare_task(focus: str = "results and metrics", **kwargs) -> Task:
    return Task(
        description=(
            "Compare all research papers gathered so far. "
            f"Focus comparison on: {focus}. "
            "Extract every quantitative metric (accuracy, F1, success rate, BLEU, etc). "
            "Structure your output as JSON in this exact format if numbers are found:\n"
            "{\n"
            '  "is_quantitative": true,\n'
            '  "metric_name": "Success Rate (%)",\n'
            '  "papers": [\n'
            '    {"label": "TD-MPC2", "value": 65},\n'
            '    {"label": "SAC", "value": 37}\n'
            '  ],\n'
            '  "context": "Compared on multi-task autonomous navigation"\n'
            "}\n"
            "If no numbers exist output: {\"is_quantitative\": false, \"summary\": \"...\"}"
        ),
        agent=analytics_agent,
        expected_output="A JSON comparison object with quantitative metrics or qualitative summary."
    )

def make_visualize_task(focus: str = "results", **kwargs) -> Task:
    return Task(
        description=(
            "You will receive the extracted content of a research paper.\n\n"
            "STEP 1: Scan for quantitative metrics (accuracy, loss, rates, numerical counts).\n"
            "STEP 2: IF NO numbers are found, or the paper is purely theoretical, STOP. "
            "Do NOT write any Python code. Instead, output EXACTLY this string: "
            "'NO_QUANTITATIVE_DATA: The paper does not contain numerical results suitable for visualization.' "
            "followed by a brief markdown summary of the qualitative findings.\n"
            "STEP 3: IF numbers ARE found, write complete, self-contained matplotlib Python code that:\n"
            "   - Hardcodes ONLY the extracted values directly in the code (never invent data).\n"
            "   - Uses the right chart type (Bar for discrete, Line for trends, Pie for parts-of-whole).\n"
            "   - Has a clear title, labeled axes, and a legend if needed.\n"
            "   - Ends with plt.tight_layout(), plt.savefig('chart.png'), plt.close().\n"
            "   - Wraps the code in ```python ... ```"
        ),
        agent=visualization_agent,
        expected_output=(
            "Either a ```python matplotlib code block that saves 'chart.png', "
            "or the exact string 'NO_QUANTITATIVE_DATA:' followed by a text summary."
        )
    )

def make_explain_task(focus: str = "all sections", **kwargs) -> Task:
    return Task(
        description=(
            "Explain the research paper concisely covering:\n"
            "1. Motivation — 2 sentences max\n"
            "2. Problem Statement — 2 sentences max\n"
            "3. Methodology — 3 sentences max\n"
            "4. Key Results — bullet points, numbers only, no prose\n"
            "5. Conclusion — 2 sentences max\n\n"
            f"Focus especially on: {focus}.\n"
            "STRICT LENGTH RULES:\n"
            "- Total response must be under 300 words\n"
            "- No introductory phrases like 'This paper presents' or 'The authors propose'\n"
            "- No filler sentences — every sentence must carry information\n"
            "- Do NOT repeat information across sections\n"
            "- Do NOT add information not present in the paper"
        ),
        agent=explainer_agent,
        expected_output=(
            "A concise structured explanation under 300 words covering "
            "motivation, problem, methodology, key results as bullet points, "
            "and conclusion. No filler, no repetition."
        )
    )

def make_concept_explain_task(topic: str = "the given concept", **kwargs) -> Task:
    return Task(
        description=(
            f"Explain this specific concept: '{topic}'.\n\n"
            "STRICT SCOPE RULE: Your entire response must be about "
            f"'{topic}' and nothing else.\n"
            "Do NOT explain parent concepts, related fields, or broader topics "
            "unless they are directly part of explaining this specific concept.\n\n"
            "Structure your response EXACTLY as:\n"
            f"Explaining: {topic}\n\n"
            "1. Definition — one precise sentence defining exactly what it is.\n"
            "2. The Problem It Solves — what specific limitation or need does it address.\n"
            "3. How It Works — the exact mechanism, step by step.\n"
            "4. Key Formula or Algorithm — if one exists, state it clearly.\n"
            "5. Concrete Example — a specific numerical or code example if possible.\n"
            "6. Variants — list specific subtypes if they exist "
            "(e.g. for attention: self-attention, cross-attention, multi-head).\n"
            "7. Limitations — what this specific concept cannot do.\n\n"
            "If you find yourself writing about topics broader than "
            f"'{topic}', stop and refocus."
        ),
        agent=concept_explainer_agent,
        expected_output=(
            f"A focused, structured explanation of '{topic}' only, "
            "covering definition, the problem it solves, mechanism, "
            "formula, example, variants, and limitations."
        )
    )


def make_find_citations_task(query: str = "the given paper", **kwargs) -> Task:
    return Task(
        description=(
            f"Find papers that cited: {query}.\n\n"
            "YOU MUST FOLLOW THESE STEPS EXACTLY IN ORDER:\n"
            "STEP 1: Call semantic_scholar_search with the query argument "
            f"set to '{query}'. Wait for the result.\n"
            "STEP 2: From the result, find the line that starts with 'PAPER_ID:'. "
            "Copy the alphanumeric string after 'PAPER_ID:'. "
            "This is your paper_id. It looks like: 204e3073870fae3d05bcbc2f6a8e263d4e58cd01\n"
            "STEP 3: Call semantic_scholar_citations with paper_id set to "
            "the exact string you copied in STEP 2. "
            "Do NOT use any other value. Do NOT use the paper title. "
            "Do NOT use placeholder text.\n"
            "STEP 4: Rank results by citation count and summarize research directions."
        ),
        agent=citation_agent,
        expected_output=(
            "A ranked list of papers that cited this work with a summary "
            "of the research directions they represent."
        )
    )


def make_find_references_task(query: str = "the given paper", **kwargs) -> Task:
    return Task(
        description=(
            f"Find papers that '{query}' builds upon.\n\n"
            "YOU MUST FOLLOW THESE STEPS EXACTLY IN ORDER:\n"
            "STEP 1: Call semantic_scholar_search with the query argument "
            f"set to '{query}'. Wait for the result.\n"
            "STEP 2: From the result, find the line that starts with 'PAPER_ID:'. "
            "Copy the alphanumeric string after 'PAPER_ID:'. "
            "This is your paper_id. It looks like: 204e3073870fae3d05bcbc2f6a8e263d4e58cd01\n"
            "STEP 3: Call semantic_scholar_references with paper_id set to "
            "the exact string you copied in STEP 2. "
            "Do NOT use any other value. Do NOT use the paper title. "
            "Do NOT use placeholder text.\n"
            "STEP 4: Rank results by citation count and summarize foundational works."
        ),
        agent=citation_agent,
        expected_output=(
            "A ranked list of papers this work references, "
            "highlighting the most foundational ones."
        )
    )



TASK_REGISTRY = {
    "search_arxiv":             make_search_task,
    "download_paper":           make_download_task,
    "read_local_file":          make_read_task,
    "summarize":                make_summarize_task,
    "generate_hypotheses":      make_hypothesis_task,
    "analyze_and_synthesize":   make_analysis_task,
    "compare_papers":           make_compare_task,      # ← new
    "visualize_results":        make_visualize_task,  
    "explain_paper":            make_explain_task, 
    "explain_concept":          make_concept_explain_task, 
    "find_citations":           make_find_citations_task,
    "find_references":          make_find_references_task
}

# knows what each task does and when to use it.
TASK_DESCRIPTIONS = {
    "search_arxiv":             "Search ArXiv for papers matching a topic or query",
    "download_paper":           "Download a specific paper by ArXiv ID or title",
    "read_local_file":          "Read and extract content from an uploaded or local PDF",
    "generate_hypotheses":      "Generate 3 testable research hypotheses from a topic",
    "analyze_and_synthesize":   "Analyze and synthesize findings ACROSS MULTIPLE papers from search results. Use this only after search_arxiv returns multiple papers",
    "compare_papers":           "Compare multiple papers by extracting quantitative metrics and results",
    "visualize_results":        "Read the Results section of a paper, extract quantitative data, and plot an appropriate chart (bar, line, or pie)",
    "explain_paper": "Explain a single research paper in full detail covering motivation, methodology, experiments, results and conclusions",  
    "explain_concept": "Explain any topic, concept, algorithm or technique from the model's own knowledge — NO paper or search needed",
    "find_citations":         "Find papers that cited a given paper and trace its research lineage forward using Semantic Scholar",
    "find_references":        "Find the foundational papers a given paper builds upon using Semantic Scholar",
}