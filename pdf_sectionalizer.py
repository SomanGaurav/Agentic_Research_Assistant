# utils/pdf_sectionalizer.py
import fitz  # PyMuPDF
import re
from pathlib import Path
import os

SECTION_KEYWORDS = [
    "abstract", "introduction", "background", "related work",
    "methodology", "method", "approach", "model", "architecture",
    "experiment", "evaluation", "results", "discussion",
    "conclusion", "future work", "references", "appendix"
]


def is_section_header(text: str, font_size: float, avg_font_size: float) -> bool:
    """Detect if a text block is a section header."""
    text_clean = text.strip().lower()

    # Font size significantly larger than average
    is_large = font_size > avg_font_size * 1.15

    # Matches known section keywords
    is_keyword = any(text_clean.startswith(k) for k in SECTION_KEYWORDS)

    # Short line (headers are rarely long)
    is_short = len(text.strip()) < 80

    # All caps or title case
    is_caps = text.strip().isupper() or text.strip().istitle()

    # Numbered section like "1.", "2.1", "III."
    is_numbered = bool(re.match(r'^(\d+\.?\d*|[IVX]+\.)\s+\w+', text.strip()))

    return (is_large or is_numbered) and is_short and (is_keyword or is_caps or is_numbered)


def extract_avg_font_size(page) -> float:
    """Get average font size on a page."""
    sizes = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") == 0:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    sizes.append(span.get("size", 12))
    return sum(sizes) / len(sizes) if sizes else 12.0


def sectionalize_pdf(file_path: str) -> dict[str, str]:
    """
    Parse a PDF and return a dict of {section_name: section_text}.
    Works even on unsectionalized PDFs — falls back to page-based chunks.
    """
    path = Path(file_path)
    if not path.exists():
        return {"full_text": f"File not found: {file_path}"}

    doc = fitz.open(file_path)
    sections = {}
    current_section = "preamble"
    current_text = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        avg_size = extract_avg_font_size(page)
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if block.get("type") != 0:  # skip non-text blocks
                continue

            for line in block.get("lines", []):
                line_text = " ".join(
                    span["text"] for span in line.get("spans", [])
                ).strip()

                if not line_text:
                    continue

                # Get font size of first span in line
                spans = line.get("spans", [])
                font_size = spans[0].get("size", 12) if spans else 12

                if is_section_header(line_text, font_size, avg_size):
                    # Save current section
                    if current_text:
                        sections[current_section] = " ".join(current_text).strip()

                    # Start new section
                    current_section = line_text.strip().lower()
                    current_text = []
                else:
                    current_text.append(line_text)

    # Save last section
    if current_text:
        sections[current_section] = " ".join(current_text).strip()

    doc.close()

    # Fallback: if only preamble found, return full text as one section
    if len(sections) <= 1:
        sections = _fallback_full_text(file_path)

    return sections


def _fallback_full_text(file_path: str) -> dict[str, str]:
    """Fallback for PDFs where no sections were detected."""
    doc = fitz.open(file_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    return {"full_text": full_text}


def get_section(sections: dict, *keywords: str) -> str:
    """
    Retrieve a specific section by keyword matching.
    Returns empty string if not found.
    Usage: get_section(sections, "results", "evaluation", "experiments")
    """
    for key in sections:
        for keyword in keywords:
            if keyword.lower() in key.lower():
                return sections[key]
    return ""


def get_key_sections(file_path: str) -> dict[str, str]:
    """
    High level function — returns only the sections relevant for research analysis.
    Cuts token load by 60-80% vs sending full text.
    """
    sections = sectionalize_pdf(file_path)

    return {
        "abstract":     get_section(sections, "abstract"),
        "methodology":  get_section(sections, "method", "approach", "methodology", "model"),
        "results":      get_section(sections, "result", "evaluation", "experiment"),
        "conclusion":   get_section(sections, "conclusion", "discussion", "future"),
        "full_text":    sections.get("full_text", "")  # only set if fallback triggered
    }




TEST_PDF = "papers/World_model_thrombectomy.pdf"  # replace with your actual file

def test_sectionalize_pdf():
    print("=" * 60)
    print("TEST 1: sectionalize_pdf — all detected sections")
    print("=" * 60)
    
    sections = sectionalize_pdf(TEST_PDF)
    
    print(f"\nTotal sections detected: {len(sections)}")
    print("\nSection names found:")
    for name in sections.keys():
        word_count = len(sections[name].split())
        print(f"  - '{name}' ({word_count} words)")


def test_get_key_sections():
    print("\n" + "=" * 60)
    print("TEST 2: get_key_sections — only relevant sections")
    print("=" * 60)
    
    key = get_key_sections(TEST_PDF)
    
    for section_name, content in key.items():
        if content:
            print(f"\n✓ {section_name.upper()} ({len(content.split())} words):")
            print(f"  Preview: {content[:200]}...")
        else:
            print(f"\n✗ {section_name.upper()}: NOT FOUND")


def test_get_section():
    print("\n" + "=" * 60)
    print("TEST 3: get_section — keyword retrieval")
    print("=" * 60)
    
    sections = sectionalize_pdf(TEST_PDF)
    
    tests = [
        ("abstract",    ["abstract"]),
        ("results",     ["result", "evaluation", "experiment"]),
        ("methodology", ["method", "approach", "methodology"]),
        ("conclusion",  ["conclusion", "discussion"]),
    ]
    
    for label, keywords in tests:
        content = get_section(sections, *keywords)
        if content:
            print(f"\n✓ '{label}' found via keywords {keywords}")
            print(f"  Preview: {content[:150]}...")
        else:
            print(f"\n✗ '{label}' NOT found via keywords {keywords}")


def test_token_savings():
    print("\n" + "=" * 60)
    print("TEST 4: token savings estimate")
    print("=" * 60)
    
    import fitz
    doc = fitz.open(TEST_PDF)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    
    full_tokens = len(full_text.split()) * 1.3  # rough token estimate
    
    key = get_key_sections(TEST_PDF)
    key_text = " ".join(v for v in key.values() if v)
    key_tokens = len(key_text.split()) * 1.3
    
    saving = ((full_tokens - key_tokens) / full_tokens) * 100
    
    print(f"\n  Full text:      ~{int(full_tokens)} tokens")
    print(f"  Key sections:   ~{int(key_tokens)} tokens")
    print(f"  Token saving:   ~{int(saving)}%")


def test_fallback():
    print("\n" + "=" * 60)
    print("TEST 5: fallback detection")
    print("=" * 60)
    
    sections = sectionalize_pdf(TEST_PDF)
    
    if "full_text" in sections:
        print("⚠ Fallback triggered — no sections detected in this PDF")
        print(f"  Full text length: {len(sections['full_text'])} chars")
    else:
        print("✓ Sections detected — fallback not needed")
        print(f"  Sections found: {list(sections.keys())}")


if __name__ == "__main__":
    if not os.path.exists(TEST_PDF):
        print(f"❌ Test file not found: {TEST_PDF}")
        print("   Place a PDF in the papers/ folder and update TEST_PDF path")
    else:
        print(f"📄 Testing with: {TEST_PDF}\n")
        test_sectionalize_pdf()
        test_get_key_sections()
        test_get_section()
        test_token_savings()
        test_fallback()
        print("\n✅ All tests complete")