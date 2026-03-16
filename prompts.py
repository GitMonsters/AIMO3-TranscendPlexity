"""
Math-specific prompts for AIMO3 competition.
Tuned for olympiad-level problems with Tool-Integrated Reasoning.
"""

SYSTEM_PROMPT = """You are an expert mathematical olympiad solver. You solve problems by writing Python code.

RULES:
1. ALWAYS write Python code to solve the problem — do NOT just reason verbally.
2. Use sympy for symbolic math, algebra, equation solving.
3. Use itertools/math for combinatorics and number theory.
4. Store your final answer in a variable called `answer`.
5. The answer MUST be a non-negative integer (0 to 99999).
6. If the problem says "find the remainder when X is divided by Y", compute X % Y.
7. If the answer could be very large, check if the problem asks for a modular result.
8. Double-check edge cases in your code.
9. Print intermediate results to verify your reasoning.

IMPORTANT: Your code must be self-contained and produce the correct integer answer.
Do NOT use any external files, network, or system commands.
"""

SOLVE_PROMPT = """Solve this math olympiad problem by writing Python code.

PROBLEM:
{problem}

Write a complete Python program that:
1. Solves this problem step by step
2. Stores the final integer answer in a variable called `answer`
3. Prints the answer at the end

Use sympy for equations/algebra if needed. Use itertools for combinatorics if needed.
Make sure `answer` is a non-negative integer.

```python
"""

RETRY_PROMPT = """Your previous attempt had an error:
{error}

Previous code:
```python
{previous_code}
```

Previous output (if any):
{stdout}

Fix the code and try again. Remember:
- Store the final answer in `answer`
- The answer must be a non-negative integer
- Use sympy for symbolic math

```python
"""

VERIFY_PROMPT = """Verify this answer to the following math problem using a DIFFERENT approach.

PROBLEM:
{problem}

CLAIMED ANSWER: {answer}

Write Python code that verifies this answer using an independent method.
If your verification confirms the answer, set `answer = {answer}`.
If your verification produces a different answer, set `answer` to your result.

```python
"""

DOMAIN_PROMPTS = {
    "algebra": """Focus on algebraic techniques:
- Factor polynomials using sympy.factor()
- Solve systems with sympy.solve()
- Check for Vieta's formulas patterns
- Consider substitutions to simplify""",

    "combinatorics": """Focus on combinatorial techniques:
- Use itertools for enumeration when feasible
- Apply inclusion-exclusion principle
- Check for bijections or recurrence relations
- Use math.comb() and math.factorial()""",

    "number_theory": """Focus on number theory techniques:
- Use modular arithmetic (pow with 3 args for fast modexp)
- Check divisibility patterns
- Apply Chinese Remainder Theorem via sympy
- Use Euler's totient, Fermat's little theorem""",

    "geometry": """Focus on geometric techniques:
- Use coordinate geometry with sympy
- Apply trigonometric identities
- Check for similar triangles, power of a point
- Use area formulas and Stewart's theorem""",
}


def classify_domain(problem: str) -> str:
    """Simple keyword-based domain classification."""
    problem_lower = problem.lower()

    geo_keywords = ["triangle", "circle", "angle", "perpendicular", "parallel",
                    "polygon", "radius", "diameter", "area", "perimeter",
                    "inscribed", "circumscribed", "tangent", "median", "altitude"]
    nt_keywords = ["divisor", "prime", "gcd", "lcm", "modulo", "remainder",
                   "divisible", "factor", "congruent", "coprime", "euler",
                   "digit", "decimal"]
    combo_keywords = ["how many", "number of ways", "permutation", "combination",
                      "arrange", "choose", "subset", "partition", "probability",
                      "expected", "sequence"]

    geo_score = sum(1 for k in geo_keywords if k in problem_lower)
    nt_score = sum(1 for k in nt_keywords if k in problem_lower)
    combo_score = sum(1 for k in combo_keywords if k in problem_lower)

    scores = {"geometry": geo_score, "number_theory": nt_score, "combinatorics": combo_score}
    best = max(scores, key=scores.get)

    if scores[best] == 0:
        return "algebra"
    return best


def build_solve_prompt(problem: str, domain_hint: bool = True) -> tuple[str, str]:
    """Build the system and user prompts for solving a problem."""
    system = SYSTEM_PROMPT
    if domain_hint:
        domain = classify_domain(problem)
        system += "\n\n" + DOMAIN_PROMPTS.get(domain, "")

    user = SOLVE_PROMPT.format(problem=problem)
    return system, user


def build_retry_prompt(problem: str, error: str, previous_code: str, stdout: str) -> tuple[str, str]:
    """Build prompts for a retry after error."""
    system = SYSTEM_PROMPT
    user = RETRY_PROMPT.format(
        error=error,
        previous_code=previous_code,
        stdout=stdout or "(no output)",
    )
    return system, user


def build_verify_prompt(problem: str, answer: int) -> tuple[str, str]:
    """Build prompts for verification with independent method."""
    system = SYSTEM_PROMPT
    user = VERIFY_PROMPT.format(problem=problem, answer=answer)
    return system, user
