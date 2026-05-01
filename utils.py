import sys
import os
import re
import uuid
import io
from typing import List
from crewai.tools import tool
import requests
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
load_dotenv()

@tool("arXiv Search")
def arxiv_search(query: str, max_results: int = 5) -> List[dict]:
    """
    Search arXiv for research papers only.

    Args:
        query: Research-focused query (e.g., 'transformer architecture NLP')
        max_results: Number of papers to return

    Returns:
        List of papers with title, authors, summary, and link
    """
    base_url = "http://export.arxiv.org/api/query"
    
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
        title = entry.find("atom:title", namespace).text.strip()
        summary = entry.find("atom:summary", namespace).text.strip()
        link = entry.find("atom:id", namespace).text.strip()

        authors = [
            author.find("atom:name", namespace).text
            for author in entry.findall("atom:author", namespace)
        ]

        papers.append({
            "title": title,
            "authors": authors,
            "summary": summary,
            "link": link
        })

    return papers

@tool("Python Plotting Executor")
def execute_plotting_code(plotting_code: str) -> str:
    """
    Executes Python matplotlib/seaborn code to generate a graph, saves it,
    and returns the saved filename.
    
    Args:
        plotting_code: A complete, self-contained Python script as a string 
                       (including necessary imports like import matplotlib.pyplot as plt) 
                       designed to generate and save a plot.
                       It must NOT contain plt.show().
                       The code MUST use plt.savefig('unique_filename.png')
    Returns:
    The path to the newly created image file if successful, or an error message.
    """
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


def get_llm_client():
    from crewai import LLM
    import os

    return LLM(
        model="gemini-2.5-flash",
        api_key=os.getenv("GEMINI_API_KEY")
    )

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

