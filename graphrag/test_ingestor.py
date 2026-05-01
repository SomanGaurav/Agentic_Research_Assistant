from pathlib import Path
# Assuming your provided code is in a file named ingestion_layer.py
from ingestor import PaperIngestor 

def run_test():
    # 1. Initialize the ingestor pointing to your PDF folder
    papers_path = Path("./test_papers")
    
    if not papers_path.exists():
        papers_path.mkdir()
        print(f"Created {papers_path}/ folder. Please add a PDF and run again.")
        return

    ingestor = PaperIngestor(papers_dir=papers_path, max_words_per_chunk=300)

    # 2. Execute ingestion
    try:
        all_chunks = ingestor.ingest_all()
    except FileNotFoundError as e:
        print(e)
        return

    # 3. Inspect the results
    print("\n--- TEST RESULTS ---")
    print(f"Total chunks created: {len(all_chunks)}")

    if all_chunks:
        # Look at the first chunk to see the metadata structure
        first = all_chunks[0]
        print(f"\nSample Chunk Data (ID: {first.chunk_id}):")
        print(f"  Paper:    {first.paper_name}")
        print(f"  Section:  {first.section} (Index: {first.section_idx})")
        print(f"  Pages:    {first.page_start} to {first.page_end}")
        print(f"  Word Count: {first.word_count}")
        print(f"  Text Snippet: {first.text[:150]}...")

        # Count chunks per section to see if regex matching worked
        from collections import Counter
        sections_found = Counter(c.section for c in all_chunks)
        print("\nChunks per Section:")
        for sec, count in sections_found.items():
            print(f"  - {sec}: {count}")

if __name__ == "__main__":
    run_test()