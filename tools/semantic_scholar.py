import re
from datetime import datetime
from crewai.tools import tool , BaseTool 
from utils import s2_get
BASE = "https://api.semanticscholar.org/graph/v1"
INTENT_WORDS = [
    "top cited", "most cited", "highly cited", "best", "top",
    "popular", "famous", "important", "seminal", "foundational",
    "key", "major", "leading", "recent", "latest", "new",
    "find me", "search for", "give me", "papers on",
    "papers about", "papers in", "research on",
    "literature on", "survey on", "overview of"
]

def clean_query(query: str) -> str:
    """Strip intent words that confuse search engines."""
    cleaned = query.lower()
    for word in INTENT_WORDS:
        cleaned = cleaned.replace(word, "")
    return " ".join(cleaned.split()).strip()


def parse_time_constraint(query: str) -> tuple[str, int | None, int | None, str]:
    """Extract year constraints from natural language."""
    clean = query
    start_year = None
    end_year = None
    desc = "no time constraint"
    current_year = datetime.now().year

    after = re.search(r'\b(?:after|since|from)\s+(20\d{2})\b', query, re.I)
    if after:
        start_year = int(after.group(1))
        clean = re.sub(after.group(0), "", clean).strip()
        desc = f"after {start_year}"

    before = re.search(r'\b(?:before|until|up to)\s+(20\d{2})\b', query, re.I)
    if before:
        end_year = int(before.group(1))
        clean = re.sub(before.group(0), "", clean).strip()
        desc = f"before {end_year}"

    in_year = re.search(r'\bin\s+(20\d{2})\b', query, re.I)
    if in_year:
        start_year = int(in_year.group(1))
        end_year = int(in_year.group(1))
        clean = re.sub(in_year.group(0), "", clean).strip()
        desc = f"in {start_year}"

    last_n = re.search(r'\blast\s+(\d+)\s+years?\b', query, re.I)
    if last_n:
        n = int(last_n.group(1))
        start_year = current_year - n
        clean = re.sub(last_n.group(0), "", clean).strip()
        desc = f"last {n} years (from {start_year})"

    between = re.search(r'\bbetween\s+(20\d{2})\s+and\s+(20\d{2})\b', query, re.I)
    if between:
        start_year = int(between.group(1))
        end_year = int(between.group(2))
        clean = re.sub(between.group(0), "", clean).strip()
        desc = f"between {start_year} and {end_year}"

    return clean.strip(), start_year, end_year, desc


@tool("s2_literature_search")
def s2_literature_search(query: str) -> str:
    """
    Search Semantic Scholar for papers ranked by citation count.
    Handles time constraints automatically from the query string.
    Use this for literature reviews and finding top cited papers.
    """
    try:
        # Step 1: clean intent words, parse time constraints
        cleaned = clean_query(query)
        clean_q, start_year, end_year, time_desc = parse_time_constraint(cleaned)

        print(f"  📚 S2 literature search: '{clean_q}' | Time: {time_desc}")

        # Step 2: build S2 query params
        params = {
            "query": clean_q,
            "limit": 20,
            "fields": "title,year,citationCount,influentialCitationCount,authors,abstract,externalIds"
        }

        # Step 3: fetch
        data = s2_get(f"{BASE}/paper/search", params=params)

        if not data.get("data"):
            return f"No papers found for: '{clean_q}'"

        papers = data["data"]

        # Step 4: apply time filter
        if start_year or end_year:
            filtered = []
            for p in papers:
                year = p.get("year")
                if not year:
                    continue
                if start_year and year < start_year:
                    continue
                if end_year and year > end_year:
                    continue
                filtered.append(p)

            if not filtered:
                return (
                    f"No papers found for '{clean_q}' with constraint: {time_desc}.\n"
                    f"Found {len(papers)} papers without the time filter. "
                    "Try relaxing the time constraint."
                )
            papers = filtered

        # Step 5: rank by citation count
        papers = sorted(
            papers,
            key=lambda p: p.get("citationCount") or 0,
            reverse=True
        )

        top = papers[:8]

        # Step 6: format output
        output = [
            f"Top Cited Papers: '{clean_q}'",
            f"Time filter: {time_desc} | Showing: {len(top)} papers\n"
        ]

        for i, p in enumerate(top, 1):
            authors = ", ".join(
                a.get("name", "") for a in p.get("authors", [])[:3]
            )
            if len(p.get("authors", [])) > 3:
                authors += " et al."

            arxiv_id = p.get("externalIds", {}).get("ArXiv", "N/A")
            paper_id = p.get("paperId", "")

            output.append(
                f"{i}. {p.get('title')}\n"
                f"   Authors: {authors}\n"
                f"   Year: {p.get('year')}\n"
                f"   Citations: {p.get('citationCount')} "
                f"| Influential: {p.get('influentialCitationCount')}\n"
                f"   ArXiv ID: {arxiv_id}\n"
                f"   S2 Paper ID: {paper_id}\n"
                f"   Abstract: {str(p.get('abstract', ''))[:200]}...\n"
            )

        return "\n".join(output)

    except Exception as e:
        return f"S2 literature search failed: {str(e)}"


@tool("s2_find_seminal")
def s2_find_seminal(topic: str) -> str:
    """
    Find seminal and foundational papers on a topic using Semantic Scholar.
    Returns older papers with high citation counts — the must-reads that
    established the field.
    """
    try:
        cleaned = clean_query(topic)
        clean_q, _, _, _ = parse_time_constraint(cleaned)

        print(f"  🏛 S2 seminal search: '{clean_q}'")

        params = {
            "query": clean_q,
            "limit": 50,
            "fields": "title,year,citationCount,influentialCitationCount,authors,abstract,externalIds"
        }

        data = s2_get(f"{BASE}/paper/search", params=params)

        if not data.get("data"):
            return f"No papers found for: '{clean_q}'"

        papers = data["data"]
        current_year = datetime.now().year

        # Seminal = high citations + at least 3 years old
        seminal = [
            p for p in papers
            if (p.get("year") or 0) <= (current_year - 3)
            and (p.get("citationCount") or 0) > 0
        ]

        if not seminal:
            seminal = papers

        # Sort by citation count descending
        seminal = sorted(
            seminal,
            key=lambda p: p.get("citationCount") or 0,
            reverse=True
        )

        top = seminal[:5]

        output = [f"Seminal Papers on '{clean_q}':\n"]

        for i, p in enumerate(top, 1):
            authors = ", ".join(
                a.get("name", "") for a in p.get("authors", [])[:3]
            )
            if len(p.get("authors", [])) > 3:
                authors += " et al."

            arxiv_id = p.get("externalIds", {}).get("ArXiv", "N/A")

            output.append(
                f"{i}. {p.get('title')}\n"
                f"   Authors: {authors}\n"
                f"   Year: {p.get('year')}\n"
                f"   Citations: {p.get('citationCount')} "
                f"| Influential: {p.get('influentialCitationCount')}\n"
                f"   ArXiv ID: {arxiv_id}\n"
                f"   S2 Paper ID: {p.get('paperId', '')}\n"
                f"   Abstract: {str(p.get('abstract', ''))[:200]}...\n"
            )

        return "\n".join(output)

    except Exception as e:
        return f"S2 seminal search failed: {str(e)}"


@tool("s2_recent_advances")
def s2_recent_advances(topic: str) -> str:
    """
    Find the most recent highly cited papers on a topic using Semantic Scholar.
    Returns papers from the last 2 years ranked by citation count.
    Use this for cutting edge and state of the art work.
    """
    try:
        cleaned = clean_query(topic)
        clean_q, _, _, _ = parse_time_constraint(cleaned)
        current_year = datetime.now().year
        cutoff = current_year - 2

        print(f"  🔬 S2 recent advances: '{clean_q}' (from {cutoff})")

        params = {
            "query": clean_q,
            "limit": 30,
            "fields": "title,year,citationCount,influentialCitationCount,authors,abstract,externalIds"
        }

        data = s2_get(f"{BASE}/paper/search", params=params)

        if not data.get("data"):
            return f"No papers found for: '{clean_q}'"

        papers = data["data"]

        # Filter to recent papers
        recent = [
            p for p in papers
            if (p.get("year") or 0) >= cutoff
        ]

        if not recent:
            return (
                f"No papers from last 2 years found for '{clean_q}'. "
                f"Most recent paper found is from "
                f"{max((p.get('year') or 0) for p in papers)}."
            )

        # Sort by citation count
        recent = sorted(
            recent,
            key=lambda p: p.get("citationCount") or 0,
            reverse=True
        )

        top = recent[:6]

        output = [f"Recent Advances on '{clean_q}' (from {cutoff}):\n"]

        for i, p in enumerate(top, 1):
            authors = ", ".join(
                a.get("name", "") for a in p.get("authors", [])[:3]
            )
            if len(p.get("authors", [])) > 3:
                authors += " et al."

            arxiv_id = p.get("externalIds", {}).get("ArXiv", "N/A")

            output.append(
                f"{i}. {p.get('title')}\n"
                f"   Authors: {authors}\n"
                f"   Year: {p.get('year')}\n"
                f"   Citations: {p.get('citationCount')} "
                f"| Influential: {p.get('influentialCitationCount')}\n"
                f"   ArXiv ID: {arxiv_id}\n"
                f"   S2 Paper ID: {p.get('paperId', '')}\n"
                f"   Abstract: {str(p.get('abstract', ''))[:200]}...\n"
            )

        return "\n".join(output)

    except Exception as e:
        return f"S2 recent advances failed: {str(e)}"