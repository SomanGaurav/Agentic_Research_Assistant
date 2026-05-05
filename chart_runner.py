import re
import os
import traceback

def extract_and_run_chart(crew_output: str) -> str | None:
    """
    Parses the visualization agent's output.
    If it contains a ```python block, extracts and executes it.
    Returns 'chart.png' path if successful, None if qualitative/failed.
    """

    # Look for ```python ... ``` block in the output
    match = re.search(r"```python(.*?)```", crew_output, re.DOTALL)

    if not match:
        # No code block found — qualitative output, nothing to run
        return None

    code = match.group(1).strip()

    # Safety check — only allow matplotlib-related code
    if "import" in code and "matplotlib" not in code:
        return None

    try:
        # Execute the extracted code in a clean namespace
        exec_globals = {}
        exec(code, exec_globals)

        if os.path.exists("chart.png"):
            return "chart.png"

    except Exception:
        traceback.print_exc()
        return None

    return None