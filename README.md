# TranscendPlexity AIMO3

**AI Mathematical Olympiad — Progress Prize 3** ($2.2M Prize Pool)

## What This Is

A Kaggle competition solver for olympiad-level math problems using **Tool-Integrated Reasoning** (TIR) — the same program-synthesis approach that got us 540/540 on ARC-AGI.

**Pipeline:**
```
Math Problem → LLM generates Python code → Execute → Extract answer
                    ↕ retry on error
              Repeat N times → Majority vote → Final answer
```

## Architecture

| Component | Purpose |
|-----------|---------|
| `notebook.py` | Kaggle submission (the actual notebook) |
| `solver.py` | Core TIR solver + majority voting |
| `sandbox.py` | Safe Python code execution with timeouts |
| `prompts.py` | Math-specific prompts with domain routing |
| `test_local.py` | Local testing against known olympiad problems |

## Model

**gpt-oss-120b** on Kaggle H100 (80GB VRAM)
- OpenAI's open-source MoE model (120B params, ~5B active per token)
- MXFP4 quantization → fits single H100
- Apache 2.0 license
- vLLM for fast inference

Also supports NVIDIA NIM endpoints and DeepSeek-R1 for development/testing.

## Quick Start

### Test sandbox locally (no LLM needed):
```bash
cd PROJECTS/aimo3
python test_local.py --sandbox-only
```

### Test with NVIDIA NIM:
```bash
export LLM_BASE_URL=https://integrate.api.nvidia.com/v1
export LLM_API_KEY=nvapi-YOUR_KEY
export LLM_MODEL=deepseek-ai/deepseek-r1
python test_local.py --problems 3 --attempts 4
```

### Test with local Ollama:
```bash
export LLM_BASE_URL=http://localhost:11434/v1
export LLM_MODEL=deepseek-r1:7b
python test_local.py
```

### Submit to Kaggle:
1. Upload `notebook.py` as a Kaggle notebook
2. Attach `openai/gpt-oss-120b` as a Kaggle model input
3. Settings: GPU H100, Internet OFF
4. Run All → Submit

## Competition Details

- **Competition:** [AIMO Progress Prize 3](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3)
- **Prize:** $2.2M total pool
- **Deadline:** April 15, 2026
- **Problems:** 110 original olympiad-level math problems
- **Hardware:** Kaggle H100 GPU, ≤5hr GPU runtime, no internet
- **Format:** Integer answers (0-99999)

## Key Techniques

1. **Tool-Integrated Reasoning** — LLM writes Python code, we execute it
2. **Majority Voting** — Run N attempts with varied temperatures, pick consensus
3. **Verification** — Independent verification attempt for low-confidence answers  
4. **Domain Routing** — Classify problem type → specialized prompts
5. **Adaptive Time Budget** — More attempts for hard problems, fewer for easy ones
6. **Retry on Error** — Feed errors back to LLM for self-correction

## By TranscendPlexity

The same team that achieved **540/540 (100%)** across all ARC-AGI benchmarks.

- 🏆 ARC-AGI-1: 400/400
- 🏆 ARC-AGI-2: 120/120  
- 🏆 ARC-AGI-3: 20/20 (interactive game solving)
- GitHub: [GitMonsters](https://github.com/GitMonsters)
