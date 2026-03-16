"""
Core AIMO3 solver with Tool-Integrated Reasoning (TIR).
Generates Python code via LLM, executes it, extracts integer answers.
Supports majority voting across multiple attempts.
"""

import re
import json
import time
import os
from collections import Counter
from typing import Optional

from sandbox import execute_code, extract_integer_answer
from prompts import build_solve_prompt, build_retry_prompt, build_verify_prompt


# ---------------------------------------------------------------------------
# LLM client — supports NVIDIA NIM, vLLM local, OpenAI-compatible endpoints
# ---------------------------------------------------------------------------

class LLMClient:
    """OpenAI-compatible client for math code generation."""

    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        model: str = None,
    ):
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "none")
        self.model = model or os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

        try:
            from openai import OpenAI
            self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        except ImportError:
            self.client = None
            print("WARNING: openai package not installed, using HTTP fallback")

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Generate a completion. Returns raw text."""
        if self.client:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        else:
            return self._http_generate(system_prompt, user_prompt, temperature, max_tokens)

    def _http_generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Fallback HTTP client when openai package unavailable."""
        import urllib.request

        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        })

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload.encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )

        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

def extract_code(text: str) -> Optional[str]:
    """Extract Python code from LLM response."""
    # Try ```python blocks first
    patterns = [
        r"```python\n(.*?)```",
        r"```\n(.*?)```",
        r"```python(.*?)```",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[-1].strip()

    # If the response IS code (no markdown), return it
    lines = text.strip().split("\n")
    code_lines = [l for l in lines if not l.startswith("#") and l.strip()]
    if code_lines and any(kw in text for kw in ["import ", "def ", "answer", "print("]):
        return text.strip()

    return None


# ---------------------------------------------------------------------------
# Single-attempt solver
# ---------------------------------------------------------------------------

def solve_single(
    client: LLMClient,
    problem: str,
    temperature: float = 0.7,
    max_retries: int = 2,
    code_timeout: int = 60,
) -> dict:
    """
    Single attempt to solve a problem via TIR.

    Returns dict with: answer (int or None), code, stdout, attempts, success
    """
    system, user = build_solve_prompt(problem)
    response = client.generate(system, user, temperature=temperature)
    code = extract_code(response)

    if not code:
        return {"answer": None, "code": None, "stdout": "", "attempts": 1, "success": False}

    # Execute code
    result = execute_code(code, timeout=code_timeout)

    # If error, retry with error context
    for retry in range(max_retries):
        if result["success"] and result["answer"] is not None:
            break

        system_r, user_r = build_retry_prompt(
            problem,
            error=result.get("error", "No answer produced"),
            previous_code=code,
            stdout=result.get("stdout", ""),
        )
        response = client.generate(system_r, user_r, temperature=temperature)
        code = extract_code(response)
        if code:
            result = execute_code(code, timeout=code_timeout)

    answer = extract_integer_answer(result.get("answer"))

    return {
        "answer": answer,
        "code": code,
        "stdout": result.get("stdout", ""),
        "attempts": 1 + retry + 1 if not result["success"] else 1,
        "success": answer is not None,
    }


# ---------------------------------------------------------------------------
# Majority-vote solver
# ---------------------------------------------------------------------------

def solve_with_voting(
    client: LLMClient,
    problem: str,
    num_attempts: int = 8,
    temperature: float = 0.7,
    code_timeout: int = 60,
    verify: bool = True,
) -> dict:
    """
    Solve a problem with majority voting across multiple attempts.

    Returns dict with: answer, confidence, attempts, all_answers
    """
    answers = []
    all_results = []

    for i in range(num_attempts):
        # Vary temperature slightly for diversity
        temp = temperature + (i * 0.05)
        temp = min(temp, 1.2)

        result = solve_single(client, problem, temperature=temp, code_timeout=code_timeout)
        all_results.append(result)

        if result["answer"] is not None:
            answers.append(result["answer"])

    if not answers:
        return {
            "answer": 0,
            "confidence": 0.0,
            "attempts": num_attempts,
            "all_answers": [],
            "verified": False,
        }

    # Majority vote
    counter = Counter(answers)
    best_answer, best_count = counter.most_common(1)[0]
    confidence = best_count / len(answers)

    # Optional verification for low-confidence answers
    verified = False
    if verify and confidence < 0.8 and len(counter) > 1:
        system_v, user_v = build_verify_prompt(problem, best_answer)
        response = client.generate(system_v, user_v, temperature=0.3)
        vcode = extract_code(response)
        if vcode:
            vresult = execute_code(vcode, timeout=code_timeout)
            vanswer = extract_integer_answer(vresult.get("answer"))
            if vanswer is not None:
                if vanswer == best_answer:
                    verified = True
                    confidence = min(confidence + 0.2, 1.0)
                elif vanswer in counter:
                    # Verification agrees with a different answer — switch
                    best_answer = vanswer
                    verified = True

    return {
        "answer": best_answer,
        "confidence": confidence,
        "attempts": num_attempts,
        "all_answers": answers,
        "verified": verified,
    }


# ---------------------------------------------------------------------------
# Batch solver (all problems)
# ---------------------------------------------------------------------------

def solve_batch(
    client: LLMClient,
    problems: list[dict],
    num_attempts: int = 8,
    time_budget_sec: int = 16200,  # 4.5 hours (leave buffer)
    code_timeout: int = 60,
) -> list[dict]:
    """
    Solve a batch of problems within a time budget.

    problems: list of {"id": str, "problem": str}
    Returns: list of {"id": str, "answer": int}
    """
    start_time = time.time()
    results = []
    n = len(problems)

    for i, prob in enumerate(problems):
        elapsed = time.time() - start_time
        remaining = time_budget_sec - elapsed

        if remaining < 60:
            # Out of time — fill remaining with 0
            print(f"[{i+1}/{n}] TIME BUDGET EXHAUSTED — filling rest with 0")
            for j in range(i, n):
                results.append({"id": problems[j]["id"], "answer": 0})
            break

        # Adaptive attempts based on remaining time
        time_per_problem = remaining / (n - i)
        adaptive_attempts = max(2, min(num_attempts, int(time_per_problem / 30)))

        print(f"[{i+1}/{n}] Solving {prob['id']} ({adaptive_attempts} attempts, {remaining:.0f}s remaining)")

        result = solve_with_voting(
            client, prob["problem"],
            num_attempts=adaptive_attempts,
            code_timeout=code_timeout,
        )

        results.append({
            "id": prob["id"],
            "answer": result["answer"],
        })

        print(f"  → Answer: {result['answer']} (confidence: {result['confidence']:.1%}, "
              f"votes: {result['all_answers']})")

    return results
