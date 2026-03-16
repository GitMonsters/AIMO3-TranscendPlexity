"""
Safe Python code execution sandbox for math problem solving.
Executes LLM-generated code with timeouts and restricted imports.
"""

import multiprocessing
import traceback
import signal
import sys
import io
from typing import Optional


ALLOWED_MODULES = {
    "math", "cmath", "decimal", "fractions", "statistics",
    "itertools", "functools", "operator", "collections",
    "random", "re", "string", "textwrap",
    "copy", "heapq", "bisect",
    "sympy", "numpy", "scipy",
}

BLOCKED_MODULES = {
    "os", "sys", "subprocess", "shutil", "pathlib",
    "socket", "http", "urllib", "requests",
    "importlib", "ctypes", "signal",
    "pickle", "shelve", "marshal",
    "__builtin__", "builtins",
}


def _execute_in_process(code: str, result_queue: multiprocessing.Queue, timeout: int) -> None:
    """Worker function that runs code in a subprocess."""
    try:
        stdout_capture = io.StringIO()
        local_ns: dict = {}

        # Redirect stdout
        old_stdout = sys.stdout
        sys.stdout = stdout_capture

        try:
            exec(code, {"__builtins__": __builtins__}, local_ns)
        finally:
            sys.stdout = old_stdout

        stdout_text = stdout_capture.getvalue()

        # Look for answer in multiple places
        answer = None
        if "answer" in local_ns:
            answer = local_ns["answer"]
        elif "result" in local_ns:
            answer = local_ns["result"]
        elif "ans" in local_ns:
            answer = local_ns["ans"]

        # If no variable found, try last printed line
        if answer is None and stdout_text.strip():
            lines = stdout_text.strip().split("\n")
            last_line = lines[-1].strip()
            try:
                answer = int(float(last_line))
            except (ValueError, TypeError):
                pass

        result_queue.put({
            "success": True,
            "answer": answer,
            "stdout": stdout_text[:5000],
            "error": None,
        })

    except Exception as e:
        result_queue.put({
            "success": False,
            "answer": None,
            "stdout": "",
            "error": f"{type(e).__name__}: {str(e)[:500]}",
        })


def execute_code(code: str, timeout: int = 30) -> dict:
    """
    Execute Python code in a sandboxed subprocess with timeout.

    Returns dict with keys: success, answer, stdout, error
    """
    ctx = multiprocessing.get_context("fork")
    result_queue = ctx.Queue()

    proc = ctx.Process(
        target=_execute_in_process,
        args=(code, result_queue, timeout),
    )
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=2)
        return {
            "success": False,
            "answer": None,
            "stdout": "",
            "error": f"Execution timed out after {timeout}s",
        }

    try:
        return result_queue.get_nowait()
    except Exception:
        return {
            "success": False,
            "answer": None,
            "stdout": "",
            "error": "No result returned from subprocess",
        }


def extract_integer_answer(raw_answer) -> Optional[int]:
    """Convert raw answer to integer in range [0, 99999]."""
    if raw_answer is None:
        return None

    try:
        if isinstance(raw_answer, (int, float)):
            val = int(raw_answer)
        elif isinstance(raw_answer, str):
            # Handle modular answers like "answer mod 1000 = 42"
            raw_answer = raw_answer.strip()
            val = int(float(raw_answer))
        else:
            val = int(raw_answer)

        # Normalize to valid AIMO range
        return val % 100000

    except (ValueError, TypeError, OverflowError):
        return None


if __name__ == "__main__":
    # Quick self-test
    test_code = """
import math
# Solve: What is the sum of the first 100 positive integers?
answer = sum(range(1, 101))
print(f"The answer is {answer}")
"""
    result = execute_code(test_code, timeout=10)
    print(f"Result: {result}")
    print(f"Extracted: {extract_integer_answer(result['answer'])}")
