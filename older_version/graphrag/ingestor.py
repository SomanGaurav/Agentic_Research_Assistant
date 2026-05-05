"""
=============================================================================
LAYER 1: INGESTION LAYER
=============================================================================
Responsibilities:
  - Read PDF files from a local directory
  - Extract clean text using PyMuPDF
  - Segment documents into semantically meaningful sections
    (Abstract, Introduction, Methods, Results, Discussion, Conclusion)
  - Output: list of structured Chunk objects with full metadata

Design Principles:
  - No dependency on upper layers (graph, retrieval, agents)
  - Each chunk is self-contained and independently addressable
  - Section detection uses heuristic regex + heading matching
  - Falls back to paragraph-level chunking if no sections detected
=============================================================================
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF


# ---------------------------------------------------------------------------
# Data Model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """
    A single semantically meaningful unit of text extracted from a paper.

    Attributes:
        chunk_id    : globally unique identifier (UUID4)
        paper_name  : stem of the source PDF file
        paper_path  : absolute path to the source PDF
        section     : detected section label (e.g. "Abstract", "Methods")
        section_idx : ordinal index of this section within the paper
        chunk_idx   : ordinal index of this chunk within the section
        text        : clean, whitespace-normalised chunk text
        page_start  : first PDF page number this chunk spans
        page_end    : last  PDF page number this chunk spans
        word_count  : number of words in `text`
    """
    chunk_id:    str
    paper_name:  str
    paper_path:  str
    section:     str
    section_idx: int
    chunk_idx:   int
    text:        str
    page_start:  int
    page_end:    int
    word_count:  int = field(init=False)

    def __post_init__(self):
        self.word_count = len(self.text.split())

    def to_dict(self) -> dict:
        return asdict(self)

    def __repr__(self) -> str:
        return (f"Chunk(id={self.chunk_id[:8]}…, paper={self.paper_name!r}, "
                f"section={self.section!r}, words={self.word_count})")


# ---------------------------------------------------------------------------
# Section Detection
# ---------------------------------------------------------------------------

# Canonical section order used for ordering and display
CANONICAL_SECTIONS = [
    "Abstract",
    "Introduction",
    "Related Work",
    "Background",
    "Methods",
    "Methodology",
    "Approach",
    "Model",
    "Architecture",
    "Experiments",
    "Experimental Setup",
    "Results",
    "Evaluation",
    "Discussion",
    "Analysis",
    "Ablation",
    "Conclusion",
    "Future Work",
    "References",
    "Appendix",
]

# Regex: matches lines that look like section headings
# Handles: "1. Introduction", "2.1 Methods", "ABSTRACT", "## Results"
_HEADING_RE = re.compile(
    r"""^
    (?:
        (?:\d+\.?)+\s+       # numbered: "1.", "2.3 "
        |
        [#]+\s*              # markdown heading: "##"
    )?
    ([A-Z][A-Za-z\s\-/&]+)  # heading text starting with capital
    \s*$
    """,
    re.VERBOSE | re.MULTILINE,
)

def _normalise(text: str) -> str:
    """Strip, collapse whitespace, remove soft hyphens and ligatures."""
    text = text.replace("\xad", "")          # soft hyphen
    text = text.replace("\ufb01", "fi")      # fi ligature
    text = text.replace("\ufb02", "fl")      # fl ligature
    text = re.sub(r"[ \t]+", " ", text)      # collapse horizontal whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)   # collapse excess blank lines
    return text.strip()


def _match_canonical_section(heading: str) -> str:
    """
    Return the best canonical section name for a detected heading, or the
    heading itself if no match is close enough.
    """
    heading_lower = heading.lower().strip()
    for canon in CANONICAL_SECTIONS:
        if canon.lower() in heading_lower or heading_lower in canon.lower():
            return canon
    return heading.strip().title()


def _extract_sections(full_text: str, page_map: List[tuple]) -> List[dict]:
    """
    Split `full_text` into sections by detecting headings.

    Returns a list of dicts:
        {section_name, text, page_start, page_end}

    `page_map` is a list of (char_offset, page_number) pairs sorted by
    char_offset, used to map text positions back to PDF pages.
    """
    def char_to_page(offset: int) -> int:
        page = 1
        for char_off, pg in page_map:
            if char_off <= offset:
                page = pg
            else:
                break
        return page

    lines = full_text.split("\n")
    sections: List[dict] = []
    current_section = "Preamble"
    current_lines: List[str] = []
    current_start_offset = 0
    running_offset = 0

    for line in lines:
        stripped = line.strip()
        match = _HEADING_RE.match(stripped) if stripped else None

        # A heading line is short (≤ 60 chars) and looks like a known section
        is_heading = (
            match is not None
            and len(stripped) <= 60
            and len(stripped) >= 3
            and any(
                kw in stripped.lower()
                for kw in [s.lower() for s in CANONICAL_SECTIONS]
                     + ["abstract", "intro", "method", "result", "discuss",
                        "conclu", "related", "experiment", "evaluat",
                        "appendix", "reference", "background", "approach",
                        "ablat", "analysis", "future"]
            )
        )

        if is_heading and current_lines:
            # Save previous section
            section_text = _normalise("\n".join(current_lines))
            if section_text:
                sections.append({
                    "section_name": _match_canonical_section(current_section),
                    "text": section_text,
                    "page_start": char_to_page(current_start_offset),
                    "page_end": char_to_page(running_offset),
                })
            current_section = stripped
            current_lines = []
            current_start_offset = running_offset
        else:
            current_lines.append(line)

        running_offset += len(line) + 1  # +1 for \n

    # Flush last section
    if current_lines:
        section_text = _normalise("\n".join(current_lines))
        if section_text:
            sections.append({
                "section_name": _match_canonical_section(current_section),
                "text": section_text,
                "page_start": char_to_page(current_start_offset),
                "page_end": char_to_page(running_offset),
            })

    return sections if sections else [{
        "section_name": "Full Text",
        "text": _normalise(full_text),
        "page_start": 1,
        "page_end": len(page_map),
    }]


# ---------------------------------------------------------------------------
# Sub-chunking
# ---------------------------------------------------------------------------

def _sub_chunk(section_text: str, max_words: int = 400) -> List[str]:
    """
    If a section is very long, split it into paragraph-aligned sub-chunks
    that do not exceed `max_words` words.  We preserve paragraph boundaries
    so sentences are never split mid-thought.
    """
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", section_text) if p.strip()]
    if not paragraphs:
        return [section_text]

    chunks: List[str] = []
    current: List[str] = []
    current_wc = 0

    for para in paragraphs:
        wc = len(para.split())
        if current_wc + wc > max_words and current:
            chunks.append("\n\n".join(current))
            current = [para]
            current_wc = wc
        else:
            current.append(para)
            current_wc += wc

    if current:
        chunks.append("\n\n".join(current))

    return chunks


# ---------------------------------------------------------------------------
# PDF Reader
# ---------------------------------------------------------------------------

def _read_pdf(pdf_path: Path) -> tuple[str, List[tuple]]:
    """
    Extract full text from a PDF using PyMuPDF.

    Returns:
        full_text : concatenated page text
        page_map  : list of (char_offset, page_number) used for page attribution
    """
    doc = fitz.open(str(pdf_path))
    full_text_parts: List[str] = []
    page_map: List[tuple] = []
    offset = 0

    for page_num, page in enumerate(doc, start=1):
        page_map.append((offset, page_num))
        text = page.get_text("text")
        full_text_parts.append(text)
        offset += len(text) + 1

    doc.close()
    return "\n".join(full_text_parts), page_map


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class PaperIngestor:
    """
    Main entry point for the Ingestion Layer.

    Usage:
        ingestor = PaperIngestor(papers_dir="./papers")
        chunks   = ingestor.ingest_all()
    """

    def __init__(self, papers_dir: str | Path, max_words_per_chunk: int = 400):
        self.papers_dir = Path(papers_dir)
        self.max_words_per_chunk = max_words_per_chunk

    # ------------------------------------------------------------------
    def ingest_all(self) -> List[Chunk]:
        """Ingest every PDF in `papers_dir` and return all chunks."""
        pdf_files = sorted(self.papers_dir.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(
                f"No PDF files found in {self.papers_dir.resolve()}"
            )

        all_chunks: List[Chunk] = []
        for pdf_path in pdf_files:
            print(f"  [Ingestor] Processing: {pdf_path.name}")
            chunks = self._ingest_paper(pdf_path)
            all_chunks.extend(chunks)
            print(f"             → {len(chunks)} chunks extracted")

        print(f"\n  [Ingestor] Total chunks across all papers: {len(all_chunks)}")
        return all_chunks

    # ------------------------------------------------------------------
    def _ingest_paper(self, pdf_path: Path) -> List[Chunk]:
        """Ingest a single PDF, returning its list of Chunk objects."""
        full_text, page_map = _read_pdf(pdf_path)
        sections = _extract_sections(full_text, page_map)

        paper_name = pdf_path.stem
        chunks: List[Chunk] = []

        for sec_idx, sec in enumerate(sections):
            sub_texts = _sub_chunk(sec["text"], self.max_words_per_chunk)

            for chunk_idx, chunk_text in enumerate(sub_texts):
                chunk = Chunk(
                    chunk_id    = str(uuid.uuid4()),
                    paper_name  = paper_name,
                    paper_path  = str(pdf_path.resolve()),
                    section     = sec["section_name"],
                    section_idx = sec_idx,
                    chunk_idx   = chunk_idx,
                    text        = chunk_text,
                    page_start  = sec["page_start"],
                    page_end    = sec["page_end"],
                )
                chunks.append(chunk)

        return chunks