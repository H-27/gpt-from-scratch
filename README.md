# GPT From Scratch

An educational / experimental repository for building language modeling components step‑by‑step:

- Text normalization (baseline + advanced configurable pipeline)
- Byte Pair Encoding (BPE) tokenizer implementation from first principles
- Vocabulary / merge rule artifacts for multiple k values (50 → 2000)
- N‑gram language model engine with Katz‑style backoff & perplexity evaluation
- Reproducible benchmarking scripts & coverage visualizations
- Jupyter report notebook summarizing methodology & results (`report_file.ipynb`)

> NOTE: You asked for `report_file.py`; the project uses a notebook named `report_file.ipynb` that presents the project, experiments, plots, and analysis.

## Quick Start

Prerequisites:
- Python >= 3.12
- (Recommended) [uv](https://github.com/astral-sh/uv) for fast dependency installs, or fallback to pip.

Install (PowerShell):
```powershell
# Using uv (preferred)
uv sync

# Or with pip
pip install -e .
```

Run tests:
```powershell
pytest -q
```

Open the report (VS Code):
1. Open `report_file.ipynb` in VS Code.
2. Execute cells sequentially to reproduce analysis.

## Project Structure

```
├─ pyproject.toml            # Project + dependencies
├─ main.py                   # Simple entry placeholder
├─ report_file.ipynb         # Comprehensive report / analysis notebook
├─ data/
│  ├─ corpora/               # Raw & cleaned text corpora (Shakespeare + 2 extra books)
│  ├─ bpe_outputs/           # Generated vocabularies & merge lists (vocab_with_k*.txt, merges_k*.txt)
│  ├─ ngram_outputs/         # Cached n‑gram & context counts (JSON)
│  ├─ emb_lm/                # Embedding / language model artifacts
│  └─ hyperparams/           # Search result JSONL + plots
├─ src/
│  ├─ tokenizer/
│  │  ├─ bpe.py              # Core BPE loop (frequency pair merging)
│  │  ├─ bpe_utils.py        # Normalization + helper utilities + coverage testing
│  │  └─ nltk_example.py     # NLTK comparison example
│  ├─ ngram_engine/
│  │  ├─ ngram.py            # NgramEngine: apply BPE, backoff distribution, sampling, perplexity
│  │  ├─ nge_utils.py        # Supporting utilities
│  │  ├─ benchmark_grid.py   # Grid / benchmark script
│  │  └─ library_bench.py    # External library benchmarking
│  └─ neural_embeddings/     # Neural embedding & GPT components
├─ tests/
│  ├─ bpe_test.py            # Coverage evaluation across k values + visualization
│  └─ ngram_test.py          # Smoke tests for sentence / word generation
└─ LICENSE
```

## Core Components

### 1. Text Normalization
Baseline (`normalize_text`) vs advanced (`advanced_normalize`) with options:
- Retain sentence punctuation / apostrophes
- Digit mapping (reduce sparsity)
- Quote / ellipsis normalization

### 2. BPE Tokenizer
File: `src/tokenizer/bpe.py`
- Pure Python frequency‑pair merging
- Saves: `vocab_with_k{K}[_adv].txt` and derived `merges_k{K}[_adv].txt`

Generate a vocabulary (example):
```python
from src.tokenizer.bpe import perform_bpe
perform_bpe(text="data/corpora/Shakespeare_clean_train.txt", k=1000, normalization="advanced")
```

### 3. Coverage / Analysis
Run the coverage + plot pipeline:
```powershell
python tests/bpe_test.py
```
Outputs plot(s) under `data/bpe_outputs/` (coverage vs k, unknown token decay).

### 4. N‑gram Engine
File: `src/ngram_engine/ngram.py`
- Loads precomputed n‑gram and context JSON caches
- Backoff probability calculation with 0.4^(order drop) weight
- Perplexity computation on validation set
- Token sampling to generate sequences.

Example usage:
```python
from src.ngram_engine.ngram import NgramEngine
engine = NgramEngine(n=3, k=1000, advanced=True)
print(engine.generate_sentence(["<s>", "the"]))
```

### 5. Report Notebook
`report_file.ipynb` consolidates:
- Data cleaning decisions
- Normalization variants & rationale
- BPE growth curves & coverage plots
- N‑gram perplexity experiments

## Milestones & Task Plan (Summary)
- Task 1: Shakespeare split, BPE vocabularies across k, normalization comparison, coverage metrics.
- Task 2: N‑gram engine (1–4 grams), add‑one smoothing, backoff, perplexity & generation.
- Subsequent tasks: Neural embeddings and GPT components integrated into codebase / notebook narrative.

### Evaluation Summary
- Coverage (%) & unknown count (tokenization phase)
- Perplexity across model stages (validation set)
- Sample generations (context → continuation)

### Mapping Tasks → Code
- BPE training & coverage: `src/tokenizer/bpe.py`, `tests/bpe_test.py`
- Normalization strategies: `src/tokenizer/bpe_utils.py`
- N‑gram model & perplexity: `src/ngram_engine/ngram.py`
- Hyperparameter artifacts: `data/hyperparams/`
- Neural / GPT components: `src/neural_embeddings/`
- Report consolidation: `report_file.ipynb`

## Typical Workflow
1. Clean / inspect corpora.
2. Generate or reuse BPE vocabularies (adjust k in `tests/bpe_test.py`).
3. Run coverage analysis to select k.
4. Build / use n‑gram statistics and compute perplexity.
5. Generate samples & iterate normalization.
6. Move to neural embeddings / transformer components as needed.

## Testing
```powershell
pytest -q
```
BPE test will (re)generate missing vocab files; n‑gram test skips gracefully if artifacts absent.

## License
See `LICENSE`.

## Citation / Attribution
Public domain / classic literature sources included for educational purposes.

---
Open `report_file.ipynb` for the narrative walkthrough and visual results.