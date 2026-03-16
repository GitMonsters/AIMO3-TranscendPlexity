#!/usr/bin/env python3
"""
TranscendPlexity AIMO3 — Kaggle Notebook Submission
AI Mathematical Olympiad Progress Prize 3 ($2.2M)

SUBMISSION FORMAT: Kaggle inference server (not batch CSV).
Uses kaggle_evaluation.aimo_3_inference_server.AIMO3InferenceServer.

Pipeline: Problem → vLLM generates Python code → sandbox executes → extract integer
          Repeat N times → majority vote → return answer

Model: gpt-oss-120b via vLLM (MXFP4 quantization, fits single H100 80GB)

Usage on Kaggle:
  1. File > Editor Type > Script
  2. Paste this entire file
  3. File > Editor Type > Notebook
  4. Add Model: openai/gpt-oss-120b as Kaggle Model input
  5. Session options > Accelerator > GPU H100
  6. Internet: OFF
  7. Save & Run All → Submit output

Author: Evan Pieser / TranscendPlexity
"""

# ===========================================================================
# CONFIGURATION
# ===========================================================================

import os
import re
import io
import sys
import time
import math
import json
import traceback
import multiprocessing
from collections import Counter
from typing import Optional

import pandas as pd
import polars as pl

# Model config — adjust path based on how you attach the model on Kaggle
MODEL_PATHS = [
    "/kaggle/input/gpt-oss-120b",
    "/kaggle/input/gpt-oss/transformers/120b/1",
    "/kaggle/input/gpt-oss/gpt-oss-120b",
    "/kaggle/input/gpt-oss/PyTorch/120b/1",
]
# Fallback models if gpt-oss isn't attached
FALLBACK_MODELS = [
    "/kaggle/input/deepseek-r1-distill-qwen-32b",
    "/kaggle/input/qwen2.5-math-72b-instruct",
    "Qwen/Qwen2.5-Math-72B-Instruct",
]

NUM_ATTEMPTS = 8           # Voting attempts per problem
CODE_TIMEOUT = 45          # Seconds per code execution
MAX_TOKENS = 4096
VLLM_GPU_MEMORY = 0.90    # Use 90% of H100 80GB
VLLM_MAX_MODEL_LEN = 8192
TOTAL_TIME_BUDGET = 16200  # 4.5 hours (buffer for 5hr limit)

# ===========================================================================
# SANDBOX — Safe Python code execution in subprocess
# ===========================================================================

def _sandbox_worker(code: str, result_queue, timeout: int):
    """Execute code in isolated subprocess."""
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

        # Extract answer from multiple sources
        answer = None
        for var in ["answer", "result", "ans", "ANSWER", "Answer"]:
            if var in local_ns:
                answer = local_ns[var]
                break

        # Fallback: last printed number
        if answer is None and stdout_text.strip():
            lines = stdout_text.strip().split("\n")
            for line in reversed(lines):
                nums = re.findall(r'-?\d+', line.strip())
                if nums:
                    try:
                        answer = int(nums[-1])
                    except (ValueError, OverflowError):
                        pass
                    break

        result_queue.put({
            "success": True,
            "answer": answer,
            "stdout": stdout_text[:5000],
            "error": None
        })
    except Exception as e:
        result_queue.put({
            "success": False,
            "answer": None,
            "stdout": "",
            "error": f"{type(e).__name__}: {str(e)[:500]}"
        })


def execute_code(code: str, timeout: int = CODE_TIMEOUT) -> dict:
    """Run code in sandboxed subprocess with hard timeout."""
    ctx = multiprocessing.get_context("fork")
    q = ctx.Queue()
    proc = ctx.Process(target=_sandbox_worker, args=(code, q, timeout))
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
    except Exception:
        return {"success": False, "answer": None, "stdout": "", "error": "No result"}


def to_int(raw) -> Optional[int]:
    """Convert raw answer to integer in AIMO range [0, 99999]."""
    if raw is None:
        return None
    try:
        if isinstance(raw, (int, float)):
            val = int(raw)
        else:
            s = str(raw).strip()
            # Handle sympy objects
            val = int(float(s))
        return val % 100000
    except (ValueError, TypeError, OverflowError):
        return None


# ===========================================================================
# PROMPTS — Math olympiad code generation
# ===========================================================================

SYSTEM_PROMPT = """You are an expert mathematical olympiad solver. You MUST solve problems by writing Python code.

RULES:
1. ALWAYS write executable Python code in a ```python block.
2. Use sympy for symbolic math, algebra, equations, number theory.
3. Use itertools/math for combinatorics.
4. Store your FINAL integer answer in a variable called `answer`.
5. The answer MUST be a non-negative integer (0 to 99999).
6. If the problem asks for a remainder mod N, compute result % N.
7. Print intermediate results to verify your reasoning.
8. Your code must be 100% self-contained. No external files or network.
9. Import everything you need at the top of your code.
10. ALWAYS end with: print(f"ANSWER: {answer}")
"""

def make_solve_prompt(problem: str) -> str:
    return f"""Solve this math olympiad problem by writing Python code.

PROBLEM:
{problem}

Write complete, self-contained Python code that:
1. Solves this step by step with clear comments
2. Uses sympy for any algebraic/symbolic work
3. Stores the final integer answer in `answer`
4. Prints: ANSWER: {{answer}}

```python
"""

def make_retry_prompt(error: str, code: str, stdout: str) -> str:
    return f"""Your previous code had an error. Fix it.

ERROR: {error}

PREVIOUS CODE:
```python
{code}
```

OUTPUT SO FAR: {stdout or '(none)'}

Write fixed Python code. Store integer answer in `answer`. Print ANSWER: {{answer}}.

```python
"""

# ===========================================================================
# CODE EXTRACTION
# ===========================================================================

def extract_code(text: str) -> Optional[str]:
    """Extract Python code from LLM response."""
    for pat in [r"```python\n(.*?)```", r"```python(.*?)```", r"```\n(.*?)```"]:
        m = re.findall(pat, text, re.DOTALL)
        if m:
            return m[-1].strip()
    # If response looks like raw code
    if any(kw in text for kw in ["import ", "def ", "answer =", "answer=", "print("]):
        return text.strip()
    return None


def extract_boxed(text: str) -> Optional[int]:
    """Extract \\boxed{} answer as fallback (no-code reasoning)."""
    matches = re.findall(r'\\boxed\{(\d+)\}', text)
    if matches:
        try:
            return int(matches[-1]) % 100000
        except ValueError:
            pass
    return None


# ===========================================================================
# LLM CLIENT — vLLM on Kaggle H100
# ===========================================================================

class VLLMSolver:
    """vLLM-based solver for Kaggle H100."""

    def __init__(self):
        self.llm = None
        self.sampling_params = None
        self.start_time = time.time()
        self.problems_solved = 0

    def load(self):
        from vllm import LLM, SamplingParams

        # Find model path — try primary paths, then fallbacks
        model_path = None
        for p in MODEL_PATHS + FALLBACK_MODELS:
            if os.path.exists(p):
                model_path = p
                print(f"Found model at: {p}")
                break

        if model_path is None:
            model_path = "openai/gpt-oss-120b"  # Fall back to HF name
            print(f"No local model found. Using HuggingFace: {model_path}")

        print(f"Loading model via vLLM...")
        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            gpu_memory_utilization=VLLM_GPU_MEMORY,
            max_model_len=VLLM_MAX_MODEL_LEN,
            quantization="mxfp4",
            dtype="auto",
            enforce_eager=True,
        )
        self.SamplingParams = SamplingParams
        print(f"Model loaded in {time.time() - self.start_time:.1f}s!")

    def generate(self, system: str, user: str, temperature: float = 0.7) -> str:
        """Generate completion via vLLM using model's native chat template."""
        params = self.SamplingParams(
            temperature=temperature,
            max_tokens=MAX_TOKENS,
            top_p=0.95,
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        # Use vLLM's chat() which auto-applies the model's chat template
        # (Harmony format for gpt-oss-120b, ChatML for Qwen, etc.)
        try:
            outputs = self.llm.chat(messages=[messages], sampling_params=params)
            return outputs[0].outputs[0].text
        except (AttributeError, TypeError):
            # Fallback: apply template manually via tokenizer
            tokenizer = self.llm.get_tokenizer()
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            outputs = self.llm.generate([prompt], params)
            return outputs[0].outputs[0].text

    def solve_single(self, problem: str, temperature: float = 0.7) -> Optional[int]:
        """Single TIR attempt: generate code → execute → extract answer."""
        try:
            resp = self.generate(SYSTEM_PROMPT, make_solve_prompt(problem), temperature)
            code = extract_code(resp)

            if not code:
                # Fallback: try to extract \boxed{} from pure reasoning
                return extract_boxed(resp)

            result = execute_code(code, timeout=CODE_TIMEOUT)

            # Retry once on error
            if not result["success"] or result["answer"] is None:
                resp2 = self.generate(
                    SYSTEM_PROMPT,
                    make_retry_prompt(
                        result.get("error", "No answer produced"),
                        code,
                        result.get("stdout", "")
                    ),
                    temperature
                )
                code2 = extract_code(resp2)
                if code2:
                    result = execute_code(code2, timeout=CODE_TIMEOUT)
                elif not result["answer"]:
                    return extract_boxed(resp2)

            return to_int(result.get("answer"))

        except Exception as e:
            print(f"    solve_single error: {e}")
            return None

    def solve_with_voting(self, problem: str, num_attempts: int = NUM_ATTEMPTS) -> int:
        """Majority-vote solver across multiple attempts."""
        answers = []

        for i in range(num_attempts):
            temp = 0.5 + (i * 0.1)  # Sweep 0.5 → 1.3
            temp = min(temp, 1.3)
            ans = self.solve_single(problem, temperature=temp)
            if ans is not None:
                answers.append(ans)
                print(f"    Attempt {i+1}/{num_attempts}: {ans}")
            else:
                print(f"    Attempt {i+1}/{num_attempts}: FAILED")

        if not answers:
            print("    All attempts failed, returning 0")
            return 0

        # Majority vote
        counter = Counter(answers)
        best, count = counter.most_common(1)[0]
        confidence = count / len(answers)
        print(f"    Vote: {best} ({confidence:.0%} confidence, {len(answers)} valid answers)")
        print(f"    Distribution: {dict(counter)}")

        return best


# ===========================================================================
# KAGGLE SUBMISSION — AIMO3InferenceServer interface
# ===========================================================================

solver = VLLMSolver()
solve_start_time = time.time()
problems_solved = 0


def predict(id_: pl.Series, problem: pl.Series) -> pl.DataFrame | pd.DataFrame:
    """
    Kaggle competition predict function.
    Called once per problem by AIMO3InferenceServer.
    """
    global problems_solved

    # Lazy-load model on first call
    if solver.llm is None:
        solver.load()

    problem_id = id_.item(0)
    problem_text: str = problem.item(0)
    problems_solved += 1

    elapsed = time.time() - solve_start_time
    remaining = TOTAL_TIME_BUDGET - elapsed

    # Adaptive attempts based on time remaining
    est_left = max(1, 110 - problems_solved + 1)
    time_per = remaining / est_left
    attempts = max(2, min(NUM_ATTEMPTS, int(time_per / 30)))

    print(f"\n{'='*60}")
    print(f"Problem {problems_solved}/110 | ID: {problem_id}")
    print(f"Attempts: {attempts} | Time left: {remaining:.0f}s | Budget/prob: {time_per:.0f}s")
    print(f"{'='*60}")
    print(f"{problem_text[:300]}...")

    try:
        answer = solver.solve_with_voting(problem_text, num_attempts=attempts)
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        traceback.print_exc()
        answer = 0

    # Ensure valid range
    answer = int(answer) % 100000

    print(f"→ FINAL ANSWER: {answer}")
    return pl.DataFrame({'id': problem_id, 'answer': answer})


# ===========================================================================
# MAIN — Kaggle inference server entry point
# ===========================================================================

import kaggle_evaluation.aimo_3_inference_server

inference_server = kaggle_evaluation.aimo_3_inference_server.AIMO3InferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    # In regular (non-competition) mode, test.csv may not exist.
    # Try local gateway with test data, fall back to a dummy run.
    test_path = '/kaggle/input/ai-mathematical-olympiad-progress-prize-3/test.csv'
    if os.path.exists(test_path):
        inference_server.run_local_gateway((test_path,))
    else:
        print("="*60)
        print("TranscendPlexity AIMO3 — Ready for Competition Submission")
        print("="*60)
        print("No test.csv found (normal for non-competition runs).")
        print("This notebook is configured for competition rerun via serve().")
        print("To submit: Save this version → Submit to Competition.")
        print()
        # Create a dummy submission so Kaggle sees output
        dummy = pl.DataFrame({'id': ['dummy'], 'answer': [0]})
        dummy.write_parquet('submission.parquet')
        print("Created dummy submission.parquet for validation.")
