# calculate_distance.py Guide

`calculate_distance.py` computes reasoning-trace distances based on the definitions in `formula.pdf`, and exports CSV files for cross-language comparisons.

## Features

- Pair mode: compute distances between two traces passed from CLI
- Batch mode: read a JSON file with multi-language traces
- In batch mode, generates one CSV per metric:
  - `Dsem`
  - `Dstr`
  - `Dord`
  - `Plen`
  - `d`
  - `Inv`

## Metric Definitions (Implemented)

- `Dsem`: semantic distance
  - Converts each step into an embedding (hashed vector)
  - Builds one-to-one minimum-cost matching with cosine distance
  - Uses average matched cost
- `Dstr`: structural distance
  - Maps each step to a structure label (`assume/derive/compute/...`)
  - Computes normalized edit distance between label sequences
- `Dord`: order distance
  - Uses semantic matches as aligned step pairs
  - Computes inversion rate (Kendall-tau style)
- `Plen`: length penalty
  - `|m-n| / max(m,n)`
- `d`: combined distance
  - `d = alpha*Dsem + beta*Dstr + gamma*Dord + lam*Plen`
- `Inv`: invariance score
  - `Inv = exp(-d)`

Default weights: `alpha=0.4, beta=0.2, gamma=0.3, lam=0.1`

## Input JSON Format

Recommended format:

```json
[
  {
    "question_id": "q1",
    "languages": {
      "zh": "1. ...\\n2. ...",
      "en": "1. ...\\n2. ...",
      "ja": "1. ...\\n2. ..."
    }
  }
]
```

The script also supports common variants automatically:
- Root object containing a `questions` list
- Per-question language map under `traces` / `variants` / `responses`
- Per-question list under `answers` / `variants` / `responses` where each item has `lang` plus `trace/reasoning/steps/...`

## Output CSV Files

Batch mode generates 6 CSV files:

- `{output_prefix}_Dsem.csv`
- `{output_prefix}_Dstr.csv`
- `{output_prefix}_Dord.csv`
- `{output_prefix}_Plen.csv`
- `{output_prefix}_d.csv`
- `{output_prefix}_Inv.csv`

Each CSV contains:
- Rows: one row per question (`question_id`)
- Columns: one column per language
- Extra column: `reference_lang` (the reference language used for that question)

Notes:
- Each language value is computed as `distance(language, reference_lang)` for that question
- For the reference language itself: `Dsem/Dstr/Dord/Plen/d = 0`, `Inv = 1`

## CLI Usage

### 1) Pair Mode

```bash
python calculate_distance.py "1. Assume ...\n2. Compute ..." "1. Suppose ...\n2. Derive ..." --pretty
```

### 2) Batch Mode (JSON -> CSV)

```bash
python calculate_distance.py \
  --input-json sample_questions.json \
  --output-dir out_csv \
  --output-prefix langdist \
  --reference-lang zh
```

## Arguments

- `--input-json`: JSON file path for batch mode
- `--output-dir`: output directory for CSV files (default `.`)
- `--output-prefix`: output filename prefix (default `distance`)
- `--reference-lang`: preferred reference language per question; if missing for a question, falls back to the first available language
- `--alpha --beta --gamma --lam`: weights for combined distance
- `--dim`: embedding dimension (default `256`)
- `--pretty`: pretty JSON output in pair mode

## Dependencies

- Python 3.9+
- Standard library only (no third-party packages)

## Notes

- Input traces are best provided as numbered steps (`1. ...`, `2. ...`)
- If numbered steps are not found, the script falls back to non-empty lines, then sentence splitting
- Embeddings are deterministic hash embeddings (reproducible, no external model), not pretrained semantic embeddings
