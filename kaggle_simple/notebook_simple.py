#!/usr/bin/env python3
"""
TranscendPlexity AIMO3 — Lightweight Kaggle Submission
Uses transformers (pre-installed on Kaggle) instead of vLLM.
Fallback notebook if gpt-oss-120b via vLLM has issues.

Model: DeepSeek-R1-Distill-Qwen-32B (or 7B as fallback)
"""

import os
import re
import io
import sys
import time
import traceback
import multiprocessing
from collections import Counter
from typing import Optional

import pandas as pd
import polars as pl
import torch

# ===========================================================================
# CONFIG
# ===========================================================================

MODEL_PATHS = [
    "/kaggle/input/deepseek-r1-distill-qwen-32b/transformers/default/1",
    "/kaggle/input/deepseek-r1-distill-qwen-32b",
    "/kaggle/input/deepseek-r1-distill-qwen-7b/transformers/default/1",
    "/kaggle/input/deepseek-r1-distill-qwen-7b",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
]

NUM_ATTEMPTS = 6
CODE_TIMEOUT = 45
MAX_TOKENS = 4096
TOTAL_TIME_BUDGET = 16200

# ===========================================================================
# SANDBOX
# ===========================================================================

def _sandbox_worker(code, result_queue, timeout):
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
        answer = None
        for var in ["answer", "result", "ans", "ANSWER", "Answer"]:
            if var in local_ns:
                answer = local_ns[var]
                break
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
        result_queue.put({"success": True, "answer": answer, "stdout": stdout_text[:5000], "error": None})
    except Exception as e:
        result_queue.put({"success": False, "answer": None, "stdout": "", "error": f"{type(e).__name__}: {str(e)[:500]}"})


def execute_code(code, timeout=CODE_TIMEOUT):
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


def to_int(raw):
    if raw is None:
        return None
    try:
        val = int(raw) if isinstance(raw, (int, float)) else int(float(str(raw).strip()))
        return val % 100000
    except (ValueError, TypeError, OverflowError):
        return None


# ===========================================================================
# PROMPTS
# ===========================================================================

SYSTEM_PROMPT = """You are an expert mathematical olympiad solver. Solve problems by writing Python code.

RULES:
1. ALWAYS write executable Python code in a ```python block.
2. Use sympy for symbolic math. Use itertools/math for combinatorics.
3. Store your final integer answer in a variable called `answer`.
4. The answer MUST be a non-negative integer (0 to 99999).
5. If the problem asks for a remainder mod N, compute result % N.
6. Print intermediate results to verify reasoning.
7. Code must be self-contained. Import everything you need.
8. End with: print(f"ANSWER: {answer}")
"""

def make_prompt(problem):
    return f"""Solve this math olympiad problem by writing Python code.

PROBLEM:
{problem}

Write complete Python code that solves this step by step.
Store the final integer answer in `answer`. Print ANSWER: {{answer}}.

```python
"""

def make_retry(error, code, stdout):
    return f"""Your previous code had an error. Fix it.
ERROR: {error}
PREVIOUS CODE:
```python
{code}
```
OUTPUT: {stdout or '(none)'}
Write fixed code. Store integer answer in `answer`.
```python
"""


def extract_code(text):
    for pat in [r"```python\n(.*?)```", r"```python(.*?)```", r"```\n(.*?)```"]:
        m = re.findall(pat, text, re.DOTALL)
        if m:
            return m[-1].strip()
    if any(kw in text for kw in ["import ", "def ", "answer =", "print("]):
        return text.strip()
    return None


def extract_boxed(text):
    matches = re.findall(r'\\boxed\{(\d+)\}', text)
    if matches:
        try:
            return int(matches[-1]) % 100000
        except ValueError:
            pass
    return None


# ===========================================================================
# MODEL — Transformers (pre-installed on Kaggle)
# ===========================================================================

class TransformersSolver:
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self.start_time = time.time()

    def load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = None
        for p in MODEL_PATHS:
            if os.path.exists(p):
                model_path = p
                print(f"Found model at: {p}")
                break
        if model_path is None:
            model_path = MODEL_PATHS[-1]
            print(f"Using HuggingFace: {model_path}")

        print(f"Loading tokenizer...")
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"Loading model...")
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        print(f"Model loaded in {time.time() - self.start_time:.1f}s!")

    def generate(self, system, user, temperature=0.7):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=MAX_TOKENS,
                temperature=max(temperature, 0.01),
                do_sample=True,
                top_p=0.95,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        return self._tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    def solve_single(self, problem, temperature=0.7):
        try:
            resp = self.generate(SYSTEM_PROMPT, make_prompt(problem), temperature)
            code = extract_code(resp)
            if not code:
                return extract_boxed(resp)
            result = execute_code(code, timeout=CODE_TIMEOUT)
            if not result["success"] or result["answer"] is None:
                resp2 = self.generate(SYSTEM_PROMPT, make_retry(
                    result.get("error", "No answer"), code, result.get("stdout", "")
                ), temperature)
                code2 = extract_code(resp2)
                if code2:
                    result = execute_code(code2, timeout=CODE_TIMEOUT)
                elif not result["answer"]:
                    return extract_boxed(resp2)
            return to_int(result.get("answer"))
        except Exception as e:
            print(f"    solve_single error: {e}")
            return None

    def solve_with_voting(self, problem, num_attempts=NUM_ATTEMPTS):
        answers = []
        for i in range(num_attempts):
            temp = 0.5 + (i * 0.12)
            ans = self.solve_single(problem, temperature=min(temp, 1.2))
            if ans is not None:
                answers.append(ans)
                print(f"    Attempt {i+1}/{num_attempts}: {ans}")
            else:
                print(f"    Attempt {i+1}/{num_attempts}: FAILED")
        if not answers:
            return 0
        counter = Counter(answers)
        best, count = counter.most_common(1)[0]
        print(f"    Vote: {best} ({count}/{len(answers)} confidence)")
        return best


# ===========================================================================
# KAGGLE SUBMISSION
# ===========================================================================

solver = TransformersSolver()
solve_start_time = time.time()
problems_solved = 0


def predict(id_: pl.Series, problem: pl.Series) -> pl.DataFrame | pd.DataFrame:
    global problems_solved
    if solver._model is None:
        solver.load()

    problem_id = id_.item(0)
    problem_text = problem.item(0)
    problems_solved += 1

    elapsed = time.time() - solve_start_time
    remaining = TOTAL_TIME_BUDGET - elapsed
    est_left = max(1, 110 - problems_solved + 1)
    time_per = remaining / est_left
    attempts = max(2, min(NUM_ATTEMPTS, int(time_per / 40)))

    print(f"\n{'='*60}")
    print(f"Problem {problems_solved}/110 | ID: {problem_id}")
    print(f"Attempts: {attempts} | Time left: {remaining:.0f}s")
    print(f"{'='*60}")

    try:
        answer = solver.solve_with_voting(problem_text, num_attempts=attempts)
    except Exception as e:
        print(f"FATAL: {e}")
        traceback.print_exc()
        answer = 0

    answer = int(answer) % 100000
    print(f"→ FINAL: {answer}")
    return pl.DataFrame({'id': problem_id, 'answer': answer})


import kaggle_evaluation.aimo_3_inference_server

inference_server = kaggle_evaluation.aimo_3_inference_server.AIMO3InferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    print("=" * 60)
    print("TranscendPlexity AIMO3 Lite — Ready for Competition")
    print("=" * 60)
    print("Save run complete. Inference happens during competition rerun.")
    dummy = pl.DataFrame({'id': ['dummy'], 'answer': [0]})
    dummy.write_parquet('submission.parquet')
    print("Created submission.parquet ✓")
