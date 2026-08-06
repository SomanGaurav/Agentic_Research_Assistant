import sys
import os
import re
import uuid
import io
from typing import List
from crewai.tools import tool , BaseTool 
import requests
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Type, Any
import json
from older_version.graphdb import Neo4jGraphDB
import fitz
import time 
load_dotenv()
import arxiv
import os
from langchain_groq import ChatGroq
from datetime import datetime
# Set your API key
os.environ["GROQ_API_KEY"] = "your_groq_api_key_here"

# Initialize the LLM
llm = ChatGroq(
    model="compound-beta",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

def parse_time_constraint(query: str) -> tuple[str, int | None, int | None, str]:
    """
    Extract year constraints from natural language query.
    Returns (clean_query, start_year, end_year, constraint_description)
    """
    clean = query
    start_year = None
    end_year = None
    desc = "no time constraint"
    current_year = datetime.now().year

    # "after 2020", "since 2021", "from 2020"
    after = re.search(r'\b(?:after|since|from)\s+(20\d{2})\b', query, re.I)
    if after:
        start_year = int(after.group(1))
        clean = re.sub(after.group(0), "", clean).strip()
        desc = f"after {start_year}"

    # "before 2022", "until 2022", "up to 2022"
    before = re.search(r'\b(?:before|until|up to)\s+(20\d{2})\b', query, re.I)
    if before:
        end_year = int(before.group(1))
        clean = re.sub(before.group(0), "", clean).strip()
        desc = f"before {end_year}"

    # "in 2022", "in the last 3 years"
    in_year = re.search(r'\bin\s+(20\d{2})\b', query, re.I)
    if in_year:
        start_year = int(in_year.group(1))
        end_year = int(in_year.group(1))
        clean = re.sub(in_year.group(0), "", clean).strip()
        desc = f"in {start_year}"

    # "last N years"
    last_n = re.search(r'\blast\s+(\d+)\s+years?\b', query, re.I)
    if last_n:
        n = int(last_n.group(1))
        start_year = current_year - n
        clean = re.sub(last_n.group(0), "", clean).strip()
        desc = f"last {n} years (from {start_year})"

    # "between 2019 and 2022"
    between = re.search(r'\bbetween\s+(20\d{2})\s+and\s+(20\d{2})\b', query, re.I)
    if between:
        start_year = int(between.group(1))
        end_year = int(between.group(2))
        clean = re.sub(between.group(0), "", clean).strip()
        desc = f"between {start_year} and {end_year}"

    return clean.strip(), start_year, end_year, desc


def filter_by_year(results: list, start_year: int | None, end_year: int | None) -> list:
    filtered = []
    for paper in results:
        year = paper.published.year
        if start_year and year < start_year:
            continue
        if end_year and year > end_year:
            continue
        filtered.append(paper)
    return filtered


@tool("arxiv_literature_search")
def arxiv_literature_search(query: str) -> str:
    """
    Perform a literature review quality search on ArXiv.
    Handles time constraints automatically from the query.
    Returns papers ranked by citation proxy (submission date + relevance).
    Use this instead of basic arxiv_search for all paper discovery tasks.
    """
    try:
        clean_query, start_year, end_year, time_desc = parse_time_constraint(query)

        print(f"Literature search: '{clean_query}' | Time: {time_desc}")

        # Fetch more than needed so we can filter and rank
        search = arxiv.Search(
            query=clean_query,
            max_results=40,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        all_results = list(search.results())

        # Apply time filter if needed
        if start_year or end_year:
            filtered = filter_by_year(all_results, start_year, end_year)
            if not filtered:
                return (
                    f"No papers found for '{clean_query}' with constraint: {time_desc}.\n"
                    f"Found {len(all_results)} papers without the time filter. "
                    f"Try relaxing the time constraint."
                )
        else:
            filtered = all_results

        if not filtered:
            return f"No papers found for: {query}"

        # Rank by recency as citation proxy
        # ArXiv doesn't expose citation counts — we use a combined score:
        # primary sort: relevance (already sorted by arxiv)
        # secondary sort: newer papers weighted slightly higher
        current_year = datetime.now().year

        def score(paper):
            age = current_year - paper.published.year
            recency_score = max(0, 10 - age)          # newer = higher score
            return recency_score

        if not (start_year or end_year):
            # No time constraint — sort by relevance + recency combined
            filtered = sorted(filtered, key=score, reverse=True)

        # Take top 8 for literature review
        top_papers = filtered[:8]

        output = [
            f"Literature Review Search Results",
            f"Query: '{clean_query}' | Time filter: {time_desc}",
            f"Papers found: {len(filtered)} | Showing top: {len(top_papers)}\n"
        ]

        for i, paper in enumerate(top_papers, 1):
            authors = ", ".join(a.name for a in paper.authors[:3])
            if len(paper.authors) > 3:
                authors += " et al."

            output.append(
                f"{i}. {paper.title}\n"
                f"   Authors: {authors}\n"
                f"   Year: {paper.published.year}\n"
                f"   ArXiv ID: {paper.entry_id.split('/')[-1]}\n"
                f"   Abstract: {paper.summary[:200]}...\n"
            )

        return "\n".join(output)

    except Exception as e:
        return f"Literature search failed: {str(e)}"


@tool("arxiv_find_seminal")
def arxiv_find_seminal(topic: str) -> str:
    """
    Find seminal and foundational papers on a topic.
    Searches for older, highly referenced works that established the field.
    Use this to find the must-read papers that everything else builds on.
    """
    try:
        clean_topic, _, _, _ = parse_time_constraint(topic)

        # Search with relevance and take older papers
        search = arxiv.Search(
            query=clean_topic,
            max_results=50,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        results = list(search.results())

        if not results:
            return f"No papers found for: {topic}"

        # Seminal papers tend to be older — filter papers > 3 years old
        current_year = datetime.now().year
        older_papers = [
            p for p in results
            if (current_year - p.published.year) >= 3
        ]

        # If no older papers, fall back to all results
        candidates = older_papers if older_papers else results

        # Sort by oldest first among top relevance results
        candidates = sorted(candidates, key=lambda p: p.published.year)

        top = candidates[:5]

        output = [
            f"Seminal Papers on '{clean_topic}':\n"
        ]

        for i, paper in enumerate(top, 1):
            authors = ", ".join(a.name for a in paper.authors[:3])
            if len(paper.authors) > 3:
                authors += " et al."

            output.append(
                f"{i}. {paper.title}\n"
                f"   Authors: {authors}\n"
                f"   Year: {paper.published.year}\n"
                f"   ArXiv ID: {paper.entry_id.split('/')[-1]}\n"
                f"   Abstract: {paper.summary[:200]}...\n"
            )

        return "\n".join(output)

    except Exception as e:
        return f"Seminal paper search failed: {str(e)}"


@tool("arxiv_recent_advances")
def arxiv_recent_advances(topic: str) -> str:
    """
    Find the most recent papers on a topic from the last 2 years.
    Use this to find cutting edge and state of the art work.
    """
    try:
        clean_topic, _, _, _ = parse_time_constraint(topic)
        current_year = datetime.now().year
        cutoff_year = current_year - 2

        search = arxiv.Search(
            query=clean_topic,
            max_results=30,
            sort_by=arxiv.SortCriterion.SubmittedDate,  # newest first
        )

        results = list(search.results())
        recent = [p for p in results if p.published.year >= cutoff_year]

        if not recent:
            return f"No recent papers (last 2 years) found for: {topic}"

        top = recent[:6]

        output = [f"Recent Advances on '{clean_topic}' (last 2 years):\n"]

        for i, paper in enumerate(top, 1):
            authors = ", ".join(a.name for a in paper.authors[:3])
            if len(paper.authors) > 3:
                authors += " et al."

            output.append(
                f"{i}. {paper.title}\n"
                f"   Authors: {authors}\n"
                f"   Year: {paper.published.year}\n"
                f"   ArXiv ID: {paper.entry_id.split('/')[-1]}\n"
                f"   Abstract: {paper.summary[:200]}...\n"
            )

        return "\n".join(output)

    except Exception as e:
        return f"Recent advances search failed: {str(e)}"
@tool("arXiv Search")
def arxiv_search(query: str, max_results: int = 5) -> List[dict]:
    """
    Search arXiv and download the PDFs to a local 'papers' folder.
    """
    base_url = "http://export.arxiv.org/api/query"
    save_dir = "papers"
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending"
    }

    response = requests.get(base_url, params=params)
    root = ET.fromstring(response.content)

    papers = []
    namespace = {"atom": "http://www.w3.org/2005/Atom"}

    for entry in root.findall("atom:entry", namespace):
        title = entry.find("atom:title", namespace).text.strip().replace('\n', ' ')
        summary = entry.find("atom:summary", namespace).text.strip()
        link = entry.find("atom:id", namespace).text.strip()
        
        # arXiv links are like http://arxiv.org/abs/2103.xxxx
        # We need the PDF version: http://arxiv.org/pdf/2103.xxxx.pdf
        pdf_url = link.replace("/abs/", "/pdf/") + ".pdf"

        authors = [
            author.find("atom:name", namespace).text
            for author in entry.findall("atom:author", namespace)
        ]

        # Sanitize filename: remove special characters and limit length
        clean_title = re.sub(r'[^\w\s-]', '', title).strip()[:50]
        filename = f"{clean_title}.pdf"
        file_path = os.path.join(save_dir, filename)

        # Download the PDF
        try:
            pdf_response = requests.get(pdf_url)
            if pdf_response.status_code == 200:
                with open(file_path, "wb") as f:
                    f.write(pdf_response.content)
                local_status = f"Downloaded to {file_path}"
            else:
                local_status = "Download failed (status code)"
        except Exception as e:
            local_status = f"Download error: {str(e)}"

        papers.append({
            "title": title,
            "authors": authors,
            "summary": summary,
            "link": link,
            "local_path": file_path,
            "status": local_status
        })

    return papers

class PlottingInput(BaseModel):
    plotting_code: str = Field(..., description="The complete, self-contained Python script to execute. MUST be properly escaped as a string.")

class PythonPlottingExecutorTool(BaseTool):
    name: str = "Python Plotting Executor"
    description: str = (
        "Executes Python matplotlib/seaborn code to generate a graph, saves it, "
        "and returns the saved filename.\n"
        "It must NOT contain plt.show().\n"
        "The code MUST use plt.savefig('unique_filename.png')"
    )
    args_schema: Type[BaseModel] = PlottingInput

    def _run(self, plotting_code: str) -> str:
        import matplotlib
        matplotlib.use('Agg') # Set backend to non-interactive so it doesn't pop up a window
        import matplotlib.pyplot as plt
        
        # Setup output directory
        output_dir = "agent_outputs/plots"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Generate unique filename
        unique_id = str(uuid.uuid4())[:8]
        filepath = os.path.join(output_dir, f"plot_{unique_id}.png")
        safe_filepath = filepath.replace("\\", "/")
        
        # Create sandboxed context for execution
        try:
            import numpy as np
            import pandas as pd
            
            exec_globals = {
                'plt': plt,
                'np': np,
                'pd': pd,
                '__builtins__': __builtins__
            }
            
            code_to_exec = plotting_code

            # Strip out ANY savefig commands the agent tried to write
            code_to_exec = re.sub(r"plt\.savefig\(.*?\)", "", plotting_code)

            # Force OUR specific save command at the very end
            code_to_exec += f"\nplt.savefig('{safe_filepath}', bbox_inches='tight')"

            # Capture stdout to prevent console spam
            stdout_capture = io.StringIO()
            sys.stdout = stdout_capture
            
            plt.clf() 
            plt.close('all')
            
            # Execute the agent's code
            exec(code_to_exec, exec_globals)
            
            sys.stdout = sys.__stdout__ # Reset stdout
            
            if os.path.exists(filepath):
                return f"Successfully generated plot. File saved at: {filepath}"
            else:
                return f"Error: Code executed but file '{filepath}' was not created."
                
        except Exception as e:
            sys.stdout = sys.__stdout__ # Reset stdout
            return f"Error during code execution: {str(e)}"
        finally:
            plt.close('all')

# Instantiate the tool here so your other files can still import 'execute_plotting_code' normally!
execute_plotting_code = PythonPlottingExecutorTool()

class GraphRAGInput(BaseModel):
    papers: str = Field(default="papers", description="The local directory path where the PDF research papers are stored.")

class GraphRAGTool(BaseTool):
    name: str = "GraphRAG Builder"
    description: str = (
        "Reads downloaded PDF files from the 'papers' directory, extracts entities "
        "and relationships, and persists them to Neo4j."
    )
    
    def _run(self, query: str = None) -> str:
        # 1. Define the path to your downloaded papers
        papers_dir = "papers"
        if not os.path.exists(papers_dir):
            return "Error: No 'papers' directory found. Run the searcher first."

        db = Neo4jGraphDB()
        pdf_files = [f for f in os.listdir(papers_dir) if f.endswith(".pdf")]
        
        if not pdf_files:
            return "No PDF files found in the papers directory."

        nodes, edges = {}, []

        for filename in pdf_files:
            file_path = os.path.join(papers_dir, filename)
            
            # 2. Extract full text from PDF
            full_text = ""
            try:
                with fitz.open(file_path) as doc:
                    for page in doc:
                        full_text += page.get_text()
            except Exception as e:
                print(f"Failed to read {filename}: {e}")
                continue

            # 3. Create the Paper Node
            # We use the filename as a proxy for the ID if metadata isn't passed
            paper_id = filename.replace(".pdf", "").replace(" ", "_")
            nodes[paper_id] = {
                "type": "paper",
                "label": filename.replace(".pdf", ""),
                "content_preview": full_text[:500] # Storing a snippet for context
            }

            # 4. Extract rich concepts from FULL TEXT
            # Use your _extract_concepts function here
            # Note: For full text, you might want to chunk it or use an LLM
            concepts = _extract_concepts(full_text) 
            
            for concept in concepts:
                c_id = concept.replace(" ", "_").lower()
                nodes.setdefault(c_id, {"type": "concept", "label": concept})
                edges.append({
                    "source": paper_id,
                    "target": c_id,
                    "relation": "discusses_in_depth",
                    "evidence": filename,
                })

        # 5. Persist to Neo4j
        graph = {"nodes": nodes, "edges": edges}
        result = db.store_graph(graph)
        summary = db.get_full_graph_summary()
        db.close()

        return f"Successfully processed {len(pdf_files)} papers.\n{result}\n\nSummary:\n{summary}"
    
class GraphQueryInput(BaseModel):
    concept: str = Field(..., description="Concept or topic to look up in the graph")

class GraphQueryTool(BaseTool):
    name: str = "Graph Query Tool"
    description: str = (
        "Queries the Neo4j knowledge graph for relationships around a given concept. "
        "Use this to find connected papers, authors, and related concepts."
    )
    args_schema: Type[BaseModel] = GraphQueryInput

    def _run(self, concept: str) -> str:
        db = Neo4jGraphDB()
        result = db.query_graph(concept)
        db.close()
        return result
    
def get_llm_client():
    from crewai import LLM
    import os

    return LLM(
        # Adding the provider name explicitly helps CrewAI bridge to LiteLLM
        model="gemini-3-flash-preview",
        api_key=os.getenv("GEMINI_API_KEY3"),
        # Optional: Add temperature or other params for better research results
        temperature=0.1 
    )

    # return LLM(
    #     model="groq/llama-3.3-70b-versatile",
    #     api_key=os.getenv("GROQ_API_KEY"),
    #     temperature=0.1
    # )

def _extract_concepts(text: str, top_n: int = 6) -> list[str]:
    """
    Lightweight concept extractor — swap this with spaCy / KeyBERT 
    / an LLM call for production-grade extraction.
    """
    import re
    # Capture noun phrases that look like technical terms (title-cased or hyphenated)
    candidates = re.findall(r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b', text)
    # Deduplicate while preserving order
    seen, result = set(), []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result[:top_n]


class LocalFolderReader(BaseTool):
    name: str = "Local Folder Reader"
    description: str = "Lists all files in a specific directory to help identify research documents or project notes."
    
    # Define the folder path as a field
    folder_path: str = Field(default="papers", description="The path to the folder containing files.")

    def _run(self, directory_path: str = None) -> str:
        # Fallback to the default path if none is provided by the agent
        path = self.folder_path
        
        try:
            if not os.path.exists(path):
                return f"Error: The directory '{path}' does not exist."
            
            files = os.listdir(path)
            if not files:
                return f"The directory '{path}' is empty."
            
            # Format the list for the agent
            file_list = "\n".join([f"- {f}" for f in files])
            return f"Files found in '{path}':\n{file_list}"
            
        except Exception as e:
            return f"An error occurred while reading the directory: {str(e)}"


def chunk_text(text, chunk_size=1200, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks
class LocalFileReader(BaseTool):
    name: str = "Local File Reader"
    description: str = (
        "Use the provided context to summarize Abstract, Methodology, and Results. "
        "from a PDF file in the research folder for high-fidelity analysis."
    )
    folder_path: str = Field(default="papers")

    def _run(self, file_name: str) -> str:
        path = os.path.join(self.folder_path, file_name)
        
        try:
            if not os.path.exists(path):
                return f"Error: File '{file_name}' not found."

            # ── Sectionalizer replaces manual regex section detection ──────────
            from pdf_sectionalizer import get_key_sections
            sections = get_key_sections(path)
            # ──────────────────────────────────────────────────────────────────

            output = "--- MINI RAG OUTPUT ---\n"

            # Sectionalized paper — use detected sections
            if not sections.get("full_text"):
                target_sections = {
                    "Abstract":    sections.get("abstract", ""),
                    "Methodology": sections.get("methodology", ""),
                    "Results":     sections.get("results", ""),
                    "Conclusion":  sections.get("conclusion", ""),
                }

                found_any = False
                for section_name, content in target_sections.items():
                    if content:
                        found_any = True
                        chunks = chunk_text(content)
                        selected_chunks = chunks[:2]
                        output += f"\n## {section_name}\n"
                        for chunk in selected_chunks:
                            output += chunk.strip() + "\n"

                if not found_any:
                    output += "No key sections detected, returning full text preview.\n"
                    output += sections.get("full_text", "")[:3000]

            # Fallback — unsectionalized paper, use capped full text
            else:
                output += "\n## Full Text (no sections detected)\n"
                chunks = chunk_text(sections["full_text"])
                for chunk in chunks[:3]:              # 3 chunks max for fallback
                    output += chunk.strip() + "\n"

            MAX_TOTAL_CHARS = 10000
            return output[:MAX_TOTAL_CHARS]

        except Exception as e:
            return f"Error reading PDF: {str(e)}"


BASE = "https://api.semanticscholar.org/graph/v1"
FIELDS = "title,year,citationCount,influentialCitationCount,authors,abstract"

def get_headers():
    api_key = os.getenv("S2_API_KEY")  # add S2_API_KEY to your .env
    if api_key:
        return {"x-api-key": api_key}
    return {}

def s2_get(url: str, params: dict, retries: int = 3) -> dict:
    """Wrapper with retry and rate limit handling."""
    for attempt in range(retries):
        try:
            r = requests.get(
                url,
                params=params,
                headers=get_headers(),
                timeout=10
            )
            if r.status_code == 429:
                wait = 2 ** attempt  # exponential backoff: 1s, 2s, 4s
                print(f"  ⚠ S2 rate limit hit, waiting {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            if attempt == retries - 1:
                raise e
            time.sleep(3)
    return {}


@tool("semantic_scholar_search")
def semantic_scholar_search(query: str) -> str:
    """
    Search for a paper on Semantic Scholar and return its S2 paper ID and metadata.
    IMPORTANT: Always extract and return the paperId field — it is needed for citations and references.
    """
    query_variants = [
        query,                                    # full query first
        " ".join(query.split()[:6]),              # first 6 words
        " ".join(query.split()[:4]),              # first 4 words
        query.split(":")[0].strip(),              # before any colon
        query.replace("-", " "),                  # remove hyphens
    ]
    # deduplicate while preserving order
    seen = set()
    query_variants = [
        q for q in query_variants
        if q and q not in seen and not seen.add(q)
    ]

    last_error = ""
    for attempt_query in query_variants:
        try:
            print(f"  🔍 Trying S2 query: '{attempt_query}'")
            data = s2_get(
                f"{BASE}/paper/search",
                params={"query": attempt_query, "limit": 3, "fields": FIELDS}
            )

            if not data.get("data"):
                last_error = f"No results for: '{attempt_query}'"
                continue

            # Pick best match — first result
            paper = data["data"][0]
            paper_id = paper.get("paperId", "")

            if not paper_id:
                last_error = "Result had no paperId"
                continue

            return (
                f"PAPER_ID: {paper_id}\n"
                f"Title: {paper.get('title')}\n"
                f"Year: {paper.get('year')}\n"
                f"Citations: {paper.get('citationCount')}\n"
                f"Influential Citations: {paper.get('influentialCitationCount')}\n"
                f"Abstract: {paper.get('abstract', 'N/A')[:300]}...\n\n"
                f"NOTE: Use PAPER_ID '{paper_id}' as the paper_id argument "
                f"for semantic_scholar_citations or semantic_scholar_references."
            )

        except Exception as e:
            last_error = str(e)
            time.sleep(2)
            continue

    return (
        f"Could not find paper after trying {len(query_variants)} query variants.\n"
        f"Last error: {last_error}\n"
        f"Queries tried: {query_variants}\n"
        "Suggestion: Try a shorter or different paper title."
    )

@tool("semantic_scholar_citations")
def semantic_scholar_citations(paper_id: str) -> str:
    """
    Given a Semantic Scholar paper ID (from semantic_scholar_search), 
    return papers that cited it.
    The paper_id must be the PAPER_ID value returned by semantic_scholar_search.
    Do NOT pass paper titles or placeholder strings as paper_id.
    """
    # Guard against agent passing wrong value
    if not paper_id or len(paper_id) < 5 or " " in paper_id:
        return (
            "Invalid paper_id provided. "
            "You must first call semantic_scholar_search to get a valid PAPER_ID, "
            "then pass that exact value here. "
            "Paper IDs are alphanumeric strings with no spaces."
        )

    try:
        data = s2_get(
            f"{BASE}/paper/{paper_id}/citations",
            params={"limit": 10, "fields": FIELDS}
        )

        if not data.get("data"):
            return "No citations found for this paper."

        results = []
        for item in data["data"]:
            p = item.get("citingPaper", {})
            results.append(
                f"- {p.get('title')} ({p.get('year')}) "
                f"| Citations: {p.get('citationCount')} "
                f"| ID: {p.get('paperId')}"
            )

        return (
            f"Papers that cited this work ({len(results)} found):\n"
            + "\n".join(results)
        )
    except Exception as e:
        return f"Citation fetch failed: {str(e)}"


@tool("semantic_scholar_references")
def semantic_scholar_references(paper_id: str) -> str:
    """
    Given a Semantic Scholar paper ID (from semantic_scholar_search),
    return papers it references.
    The paper_id must be the PAPER_ID value returned by semantic_scholar_search.
    Do NOT pass paper titles or placeholder strings as paper_id.
    """
    # Guard against agent passing wrong value
    if not paper_id or len(paper_id) < 5 or " " in paper_id:
        return (
            "Invalid paper_id provided. "
            "You must first call semantic_scholar_search to get a valid PAPER_ID, "
            "then pass that exact value here. "
            "Paper IDs are alphanumeric strings with no spaces."
        )

    try:
        data = s2_get(
            f"{BASE}/paper/{paper_id}/references",
            params={"limit": 10, "fields": FIELDS}
        )

        if not data.get("data"):
            return "No references found for this paper."

        results = []
        for item in data["data"]:
            p = item.get("citedPaper", {})
            results.append(
                f"- {p.get('title')} ({p.get('year')}) "
                f"| Citations: {p.get('citationCount')} "
                f"| ID: {p.get('paperId')}"
            )

        return (
            f"Papers referenced by this work ({len(results)} found):\n"
            + "\n".join(results)
        )
    except Exception as e:
        return f"References fetch failed: {str(e)}"

if __name__ == "__main__":
    which_test = int(input("Enter 1 for arXiv search test, 2 for Gemini API test , 3 for Plotting test: "))
    if which_test == 1:
        query = "abs:transformer AND abs:nlp AND abs:attention"
        results = arxiv_search(query)
        for idx, paper in enumerate(results, 1):
            print(f"{idx}. {paper['title']} by {', '.join(paper['authors'])}")
            print(f"   Summary: {paper['summary']}")
            print(f"   Link: {paper['link']}\n")
    elif which_test == 2:
        from google import genai
        from dotenv import load_dotenv
        # Automatically picks up GEMINI_API_KEY from environment
        load_dotenv()
        client = genai.Client()
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents="Explain how AI works in a few words"
        )
        print(response.text)
    elif which_test == 3:
        # Test code for the agent to execute
        test_code = """
        import numpy as np
        import matplotlib.pyplot as plt
        
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        
        plt.plot(x, y, color='blue', label='Sine Wave')
        plt.title('Agent Generated Plot Test')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.legend()
        # The tool will automatically append the savefig command
        """
        print("\nExecuting test plot code...")
        result = execute_plotting_code(test_code)
        print(result)

