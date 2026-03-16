#!/usr/bin/env python3
"""
TranscendPlexity AIMO3 — Kaggle Notebook Submission
AI Mathematical Olympiad Progress Prize 3 ($2.2M)

This notebook runs on Kaggle H100 GPU with internet OFF.
It loads gpt-oss-120b via vLLM and solves 110 olympiad math problems
using Tool-Integrated Reasoning (code generation + execution + majority voting).

Usage on Kaggle:
  1. Upload this as a Kaggle notebook
  2. Attach gpt-oss-120b model as a Kaggle dataset input
  3. Set Accelerator: GPU H100
  4. Set Internet: OFF
  5. Run All → Submit

Author: Evan Pieser / TranscendPlexity
"""

# ===========================================================================
# CONFIGURATION
# ===========================================================================

MODEL_PATH = "/kaggle/input/gpt-oss-120b"  # Kaggle dataset path
MODEL_NAME = "openai/gpt-oss-120b"
NUM_ATTEMPTS = 8           # Voting attempts per problem
CODE_TIMEOUT = 60          # Seconds per code execution
TIME_BUDGET = 16200        # 4.5 hours total (buffer for 5hr limit)
MAX_TOKENS = 4096
VLLM_GPU_MEMORY = 0.92    # Use 92% of H100 80GB
VLLM_MAX_MODEL_LEN = 8192

# ===========================================================================
# INSTALL DEPENDENCIES (cached in Kaggle dataset if needed)
# ===========================================================================

import subprocess
import sys

def install_if_needed(package: str, import_name: str = None):
    try:
        __import__(import_name or package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

install_if_needed("vllm")
install_if_needed("sympy")
install_if_needed("openai")

# ===========================================================================
# IMPORTS
# ===========================================================================

import os
import re
import json
import time
import math
import io
import multiprocessing
from collections import Counter
from typing import Optional
import pandas as pd

# ===========================================================================
# SANDBOX — Safe code execution
# ===========================================================================

def _execute_worker(code: str, result_queue, timeout: int):
    """Execute code in subprocess."""
    try:
        stdout_buf = io.StringIO()
        local_ns = {}
        old_stdout = sys.stdout
        sys.stdout = stdout_buf
        try:
            exec(code, {"__builtins__": __builtins__}, local_ns)
        finally:
            sys.stdout = old_stdout

        stdout_text = stdout_buf.getvalue()
        answer = local_ns.get("answer") or local_ns.get("result") or local_ns.get("ans")

        if answer is None and stdout_text.strip():
            last = stdout_text.strip().split("\n")[-1].strip()
            try:
                answer = int(float(last))
            except (ValueError, TypeError):
                pass

        result_queue.put({"success": True, "answer": answer, "stdout": stdout_text[:5000], "error": None})
    except Exception as e:
        result_queue.put({"success": False, "answer": None, "stdout": "", "error": f"{type(e).__name__}: {str(e)[:500]}"})


def execute_code(code: str, timeout: int = 60) -> dict:
    ctx = multiprocessing.get_context("fork")
    q = ctx.Queue()
    proc = ctx.Process(target=_execute_worker, args=(code, q, timeout))
    proc.start()
    proc.join(timeout=timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()
            proc.join(2)
        return {"success": False, "answer": None, "stdout": "", "error": f"Timeout ({timeout}s)"}
    try:
        return q.get_nowait()
    except:
        return {"success": False, "answer": None, "stdout": "", "error": "No result"}


def extract_integer(raw) -> Optional[int]:
    if raw is None:
        return None
    try:
        val = int(float(str(raw).strip())) if not isinstance(raw, (int, float)) else int(raw)
        return val % 100000
    except:
        return None

# ===========================================================================
# PROMPTS
# ===========================================================================

SYSTEM = """You are an expert mathematical olympiad solver. Solve problems by writing Python code.

RULES:
1. ALWAYS write Python code — do NOT just reason verbally.
2. Use sympy for symbolic math, algebra, equation solving.
3. Use itertools/math for combinatorics and number theory.
4. Store your final answer in a variable called `answer`.
5. The answer MUST be a non-negative integer (0 to 99999).
6. If the problem says "find the remainder when X is divided by Y", compute X % Y.
7. Print intermediate results to verify your reasoning.
8. Your code must be self-contained. No external files or network.
"""

def make_prompt(problem: str) -> str:
    return f"""Solve this math olympiad problem by writing Python code.

PROBLEM:
{problem}

Write a complete, self-contained Python program that solves this step by step.
Store the final integer answer in `answer`. Print the answer.

```python
"""

def make_retry_prompt(error: str, code: str, stdout: str) -> str:
    return f"""Your previous attempt had an error:
{error}

Previous code:
```python
{code}
```
Output: {stdout or '(none)'}

Fix the code. Store integer answer in `answer`.

```python
"""

def make_verify_prompt(problem: str, answer: int) -> str:
    return f"""Verify this answer using a DIFFERENT method.

PROBLEM: {problem}
CLAIMED ANSWER: {answer}

If correct, set answer = {answer}. If wrong, set answer to your result.

```python
"""

# ===========================================================================
# CODE EXTRACTION
# ===========================================================================

def extract_code(text: str) -> Optional[str]:
    for pat in [r"```python\n(.*?)```", r"```\n(.*?)```", r"```python(.*?)```"]:
        m = re.findall(pat, text, re.DOTALL)
        if m:
            return m[-1].strip()
    if any(kw in text for kw in ["import ", "def ", "answer", "print("]):
        return text.strip()
    return None

# ===========================================================================
# LLM CLIENT — vLLM local on H100
# ===========================================================================

class VLLMClient:
    """Local vLLM inference on Kaggle H100."""

    def __init__(self, model_path: str):
        from vllm import LLM, SamplingParams
        print(f"Loading model from {model_path}...")
        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            gpu_memory_utilization=VLLM_GPU_MEMORY,
            max_model_len=VLLM_MAX_MODEL_LEN,
            quantization="mxfp4",
            dtype="auto",
        )
        self.SamplingParams = SamplingParams
        print("Model loaded!")

    def generate(self, system: str, user: str, temperature: float = 0.7, max_tokens: int = 4096) -> str:
        params = self.SamplingParams(temperature=temperature, max_tokens=max_tokens)
        prompt = f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"
        outputs = self.llm.generate([prompt], params)
        return outputs[0].outputs[0].text


class OpenAIClient:
    """OpenAI-compatible client for NVIDIA NIM or remote vLLM."""

    def __init__(self, base_url: str, api_key: str, model: str):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def generate(self, system: str, user: str, temperature: float = 0.7, max_tokens: int = 4096) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content

# ===========================================================================
# SOLVER
# ===========================================================================

def solve_single(client, problem: str, temperature: float = 0.7) -> Optional[int]:
    """Single TIR attempt: prompt → code → execute → answer."""
    try:
        resp = client.generate(SYSTEM, make_prompt(problem), temperature=temperature)
        code = extract_code(resp)
        if not code:
            return None

        result = execute_code(code, timeout=CODE_TIMEOUT)

        # Retry on error
        if not result["success"] or result["answer"] is None:
            resp2 = client.generate(SYSTEM, make_retry_prompt(
                result.get("error", "No answer"), code, result.get("stdout", "")
            ), temperature=temperature)
            code2 = extract_code(resp2)
            if code2:
                result = execute_code(code2, timeout=CODE_TIMEOUT)

        return extract_integer(result.get("answer"))
    except Exception as e:
        print(f"  solve_single error: {e}")
        return None


def solve_with_voting(client, problem: str, num_attempts: int = 8) -> tuple[int, float]:
    """Majority-vote solver. Returns (answer, confidence)."""
    answers = []
    for i in range(num_attempts):
        temp = 0.6 + (i * 0.08)
        ans = solve_single(client, problem, temperature=min(temp, 1.2))
        if ans is not None:
            answers.append(ans)

    if not answers:
        return 0, 0.0

    counter = Counter(answers)
    best, count = counter.most_common(1)[0]
    conf = count / len(answers)

    # Verify low-confidence answers
    if conf < 0.6 and len(counter) > 1:
        try:
            resp = client.generate(SYSTEM, make_verify_prompt(problem, best), temperature=0.3)
            code = extract_code(resp)
            if code:
                res = execute_code(code, timeout=CODE_TIMEOUT)
                vans = extract_integer(res.get("answer"))
                if vans is not None and vans in counter:
                    best = vans
                    conf = min(conf + 0.2, 1.0)
        except:
            pass

    return best, conf

# ===========================================================================
# KAGGLE MODEL CLASS (required submission interface)
# ===========================================================================

class Model:
    """AIMO3 Kaggle submission interface."""

    def __init__(self):
        # Try local vLLM first (Kaggle H100), fall back to remote
        if os.path.exists(MODEL_PATH):
            print("Using local vLLM on H100")
            self.client = VLLMClient(MODEL_PATH)
        elif os.getenv("LLM_BASE_URL"):
            print("Using remote endpoint")
            self.client = OpenAIClient(
                os.getenv("LLM_BASE_URL"),
                os.getenv("LLM_API_KEY", "none"),
                os.getenv("LLM_MODEL", MODEL_NAME),
            )
        else:
            raise RuntimeError("No model available! Set LLM_BASE_URL or attach model dataset.")

        self.start_time = time.time()
        self.problems_solved = 0

    def query(self, problem_statement: str) -> int:
        """Solve a single problem. Returns integer answer."""
        self.problems_solved += 1
        elapsed = time.time() - self.start_time
        remaining = TIME_BUDGET - elapsed

        # Adaptive attempts based on remaining time
        est_problems_left = max(1, 110 - self.problems_solved + 1)
        time_per_problem = remaining / est_problems_left
        attempts = max(2, min(NUM_ATTEMPTS, int(time_per_problem / 25)))

        print(f"\n{'='*60}")
        print(f"Problem {self.problems_solved}/110 | {attempts} attempts | {remaining:.0f}s remaining")
        print(f"{'='*60}")
        print(f"{problem_statement[:200]}...")

        answer, conf = solve_with_voting(self.client, problem_statement, num_attempts=attempts)

        print(f"→ Answer: {answer} (confidence: {conf:.0%})")
        return answer


# ===========================================================================
# MAIN — runs when notebook is executed
# ===========================================================================

if __name__ == "__main__":
    import glob

    # Check if we're on Kaggle
    test_csv = "/kaggle/input/ai-mathematical-olympiad-progress-prize-3/test.csv"

    if os.path.exists(test_csv):
        # --- KAGGLE MODE ---
        print("=" * 60)
        print("TranscendPlexity AIMO3 — Kaggle Submission")
        print("=" * 60)

        df = pd.read_csv(test_csv)
        model = Model()
        answers = []

        for _, row in df.iterrows():
            ans = model.query(row["problem"])
            answers.append({"id": row["id"], "answer": ans})

        result_df = pd.DataFrame(answers)
        result_df.to_csv("submission.csv", index=False)
        result_df.to_parquet("submission.parquet", index=False)
        print(f"\nDone! {len(answers)} problems solved.")
        print(result_df.head(10))

    else:
        # --- LOCAL TEST MODE ---
        print("=" * 60)
        print("TranscendPlexity AIMO3 — Local Test Mode")
        print("=" * 60)
        print("No Kaggle test.csv found. Run test_local.py for local testing.")
