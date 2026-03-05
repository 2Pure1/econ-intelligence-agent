"""
agent/tools/calculator.py
-------------------------
Python REPL tool for the Econ Intelligence Agent.
Provides a sandboxed-ish environment for statistical and mathematical computations.
"""

import sys
import io
import contextlib
from typing import Any
from loguru import logger

# Import common math/stats libs as requested in agent.py
try:
    import numpy as np
    import pandas as pd
    import scipy.stats as stats
except ImportError:
    logger.warning("Optional dependencies for calculator (numpy, pandas, scipy) not found.")

def calculate(code: str, description: str) -> str:
    """
    Execute a Python expression or small script and return the printed output.
    
    Args:
        code: The Python code to execute. Must print the result.
        description: A human-readable description of the calculation.
    """
    logger.info(f"Executing calculation: {description}")
    
    # We use a shared namespace with common libs pre-imported
    namespace = {
        "np": sys.modules.get("numpy"),
        "pd": sys.modules.get("pandas"),
        "stats": sys.modules.get("scipy.stats"),
        "math": __import__("math"),
    }
    
    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            # We use exec for scripts, or eval if it's a single expression (but exec handles both if they print)
            exec(code, namespace)
        
        result = stdout.getvalue().strip()
        if not result:
            # If nothing was printed, try to eval the last line or common patterns
            return "Calculation completed. Ensure your code use 'print()' to show results."
        return result
    except Exception as e:
        logger.error(f"Calculation failed: {e}")
        return f"Error executing calculation: {str(e)}"
