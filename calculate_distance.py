#!/usr/bin/env python3
"""Compute reasoning-trace distances and export per-metric CSV tables."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
from collections import Counter
from hashlib import md5
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

METRICS = ("Dsem", "Dstr", "Dord", "Plen", "d", "Inv")


def parse_steps(text: str) -> List[str]:
    """Parse numbered steps (e.g., '1. xxx') from text, with robust fallbacks."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    numbered: List[str] = []
    pattern = re.compile(r"^\s*\d+\s*[\.\)、:]\s*(.+)$")
    for line in lines:
        m = pattern.match(line)
        if m:
            content = m.group(1).strip()
            if content:
                numbered.append(content)

    if numbered:
        return numbered

    if lines:
        return lines

    sentence_parts = [p.strip() for p in re.split(r"[。！？!?;；]+", text) if p.strip()]
    return sentence_parts


def _tokenize(text: str) -> List[str]:
    """Mixed tokenizer for English words + CJK characters."""
    en_words = re.findall(r"[A-Za-z0-9_]+", text.lower())
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
    return en_words + cjk_chars


def embed_step(text: str, dim: int = 256) -> List[float]:
    """Create a deterministic hashed embedding vector for one reasoning step."""
    vec = [0.0] * dim
    tokens = _tokenize(text)
    if not tokens:
        return vec

    counts = Counter(tokens)
    for token, weight in counts.items():
        digest = md5(token.encode("utf-8")).hexdigest()
        idx = int(digest[:8], 16) % dim
        sign = -1.0 if (int(digest[8:10], 16) % 2) else 1.0
        vec[idx] += sign * float(weight)

    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def embed_steps(steps: Sequence[str], dim: int = 256) -> List[List[float]]:
    return [embed_step(step, dim=dim) for step in steps]


def cosine_distance(v1: Sequence[float], v2: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    if n1 == 0.0 and n2 == 0.0:
        return 0.0
    if n1 == 0.0 or n2 == 0.0:
        return 1.0
    cos = dot / (n1 * n2)
    cos = max(-1.0, min(1.0, cos))
    return 1.0 - cos


def _exact_min_assignment(cost_matrix: List[List[float]]) -> Tuple[float, List[Tuple[int, int]]]:
    """Exact minimum assignment with DP over bitmasks."""
    n_short = len(cost_matrix)

    if n_short == 0:
        return 0.0, []

    dp: Dict[int, Tuple[float, List[Tuple[int, int]]]] = {0: (0.0, [])}

    for i in range(n_short):
        next_dp: Dict[int, Tuple[float, List[Tuple[int, int]]]] = {}
        for mask, (cur_cost, pairs) in dp.items():
            for j in range(len(cost_matrix[i])):
                if mask & (1 << j):
                    continue
                nmask = mask | (1 << j)
                c = cur_cost + cost_matrix[i][j]
                prev = next_dp.get(nmask)
                if prev is None or c < prev[0]:
                    next_dp[nmask] = (c, pairs + [(i, j)])
        dp = next_dp

    best_cost = math.inf
    best_pairs: List[Tuple[int, int]] = []
    for _, (cost, pairs) in dp.items():
        if cost < best_cost:
            best_cost = cost
            best_pairs = pairs
    return best_cost, best_pairs


def _greedy_min_assignment(cost_matrix: List[List[float]]) -> Tuple[float, List[Tuple[int, int]]]:
    """Greedy approximation for large matrices."""
    n_short = len(cost_matrix)
    n_long = len(cost_matrix[0]) if n_short else 0
    used = set()
    total = 0.0
    pairs: List[Tuple[int, int]] = []
    for i in range(n_short):
        best_j = None
        best_cost = math.inf
        for j in range(n_long):
            if j in used:
                continue
            c = cost_matrix[i][j]
            if c < best_cost:
                best_cost = c
                best_j = j
        if best_j is None:
            break
        used.add(best_j)
        total += best_cost
        pairs.append((i, best_j))
    return total, pairs


def minimum_assignment(cost_matrix: List[List[float]], exact_limit: int = 14) -> Tuple[float, List[Tuple[int, int]]]:
    n_short = len(cost_matrix)
    n_long = len(cost_matrix[0]) if n_short else 0
    if n_short == 0 or n_long == 0:
        return 0.0, []

    if n_long <= exact_limit:
        return _exact_min_assignment(cost_matrix)
    return _greedy_min_assignment(cost_matrix)


def semantic_distance(s_steps: Sequence[str], t_steps: Sequence[str], dim: int = 256) -> Tuple[float, List[Tuple[int, int]]]:
    """Dsem(S,T) = min over one-to-one matching of average cosine distance."""
    if not s_steps and not t_steps:
        return 0.0, []
    if not s_steps or not t_steps:
        return 1.0, []

    s_emb = embed_steps(s_steps, dim=dim)
    t_emb = embed_steps(t_steps, dim=dim)

    if len(s_steps) <= len(t_steps):
        short_emb, long_emb, swap = s_emb, t_emb, False
    else:
        short_emb, long_emb, swap = t_emb, s_emb, True

    cost_matrix = [[cosine_distance(e1, e2) for e2 in long_emb] for e1 in short_emb]
    min_cost, pairs = minimum_assignment(cost_matrix)

    aligned_pairs = [(j, k) if not swap else (k, j) for j, k in pairs]
    dsem = min_cost / min(len(s_steps), len(t_steps))
    return dsem, aligned_pairs


def _structure_label(step: str) -> str:
    s = step.lower()
    rules = [
        ("assume", ["假设", "设", "assume", "suppose", "given"]),
        ("derive", ["推导", "因此", "所以", "故", "derive", "thus", "therefore"]),
        ("compute", ["计算", "求", "代入", "compute", "calculate", "evaluate"]),
        ("compare", ["比较", "对比", "greater", "less", "compare", "than"]),
        ("conclude", ["结论", "综上", "最终", "conclude", "answer", "hence"]),
        ("question", ["问", "求证", "prove", "show", "why"]),
    ]
    for label, kws in rules:
        if any(kw in s for kw in kws):
            return label
    return "other"


def structure_sequence(steps: Sequence[str]) -> List[str]:
    return [_structure_label(step) for step in steps]


def _edit_distance(seq1: Sequence[str], seq2: Sequence[str]) -> int:
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]


def structural_distance(s_steps: Sequence[str], t_steps: Sequence[str]) -> float:
    """Dstr(S,T) = EditDist(sigma(S), sigma(T)) / max(m,n)."""
    m, n = len(s_steps), len(t_steps)
    if m == 0 and n == 0:
        return 0.0
    s_labels = structure_sequence(s_steps)
    t_labels = structure_sequence(t_steps)
    return _edit_distance(s_labels, t_labels) / max(m, n)


def order_distance(s_steps: Sequence[str], t_steps: Sequence[str], dim: int = 256) -> float:
    """Dord via inversion rate on semantically aligned step pairs."""
    _, matches = semantic_distance(s_steps, t_steps, dim=dim)
    if len(matches) < 2:
        return 0.0

    inversions = 0
    total_pairs = 0
    for (j1, k1), (j2, k2) in itertools.combinations(matches, 2):
        total_pairs += 1
        if (j1 - j2) * (k1 - k2) < 0:
            inversions += 1

    if total_pairs == 0:
        return 0.0
    return inversions / total_pairs


def length_penalty(s_steps: Sequence[str], t_steps: Sequence[str]) -> float:
    """Plen(S,T) = |m-n| / max(m,n)."""
    m, n = len(s_steps), len(t_steps)
    if m == 0 and n == 0:
        return 0.0
    return abs(m - n) / max(m, n)


def combined_distance(
    s_steps: Sequence[str],
    t_steps: Sequence[str],
    alpha: float = 0.4,
    beta: float = 0.2,
    gamma: float = 0.3,
    lam: float = 0.1,
    dim: int = 256,
) -> Dict[str, float]:
    d_sem, _ = semantic_distance(s_steps, t_steps, dim=dim)
    d_str = structural_distance(s_steps, t_steps)
    d_ord = order_distance(s_steps, t_steps, dim=dim)
    p_len = length_penalty(s_steps, t_steps)

    d = alpha * d_sem + beta * d_str + gamma * d_ord + lam * p_len
    inv = math.exp(-d)

    return {
        "Dsem": d_sem,
        "Dstr": d_str,
        "Dord": d_ord,
        "Plen": p_len,
        "d": d,
        "Inv": inv,
    }


def calculate_trace_distance(
    trace_a: str,
    trace_b: str,
    alpha: float = 0.4,
    beta: float = 0.2,
    gamma: float = 0.3,
    lam: float = 0.1,
    dim: int = 256,
) -> Dict[str, object]:
    s_steps = parse_steps(trace_a)
    t_steps = parse_steps(trace_b)

    dist = combined_distance(s_steps, t_steps, alpha=alpha, beta=beta, gamma=gamma, lam=lam, dim=dim)
    _, matches = semantic_distance(s_steps, t_steps, dim=dim)

    return {"steps_a": s_steps, "steps_b": t_steps, "matches": matches, **dist}


def _trace_from_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                parts.append(str(item.get("text") or item.get("step") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    if isinstance(value, Mapping):
        for key in ("trace", "reasoning", "steps", "cot", "content", "text"):
            if key in value:
                return _trace_from_value(value[key])
    return str(value)


def _extract_lang_and_trace(item: Mapping[str, object]) -> Tuple[str | None, str]:
    lang = None
    for k in ("lang", "language", "locale"):
        if k in item and item[k]:
            lang = str(item[k])
            break
    trace = ""
    for k in ("trace", "reasoning", "steps", "cot", "content", "text", "answer"):
        if k in item:
            trace = _trace_from_value(item[k])
            break
    return lang, trace


def _extract_language_traces(question_obj: Mapping[str, object]) -> Dict[str, str]:
    for key in ("languages", "traces", "variants", "responses"):
        v = question_obj.get(key)
        if isinstance(v, Mapping):
            out: Dict[str, str] = {}
            for lang, trace_value in v.items():
                out[str(lang)] = _trace_from_value(trace_value)
            return out

    for key in ("answers", "variants", "responses", "language_traces"):
        v = question_obj.get(key)
        if isinstance(v, list):
            out: Dict[str, str] = {}
            for item in v:
                if isinstance(item, Mapping):
                    lang, trace = _extract_lang_and_trace(item)
                    if lang:
                        out[lang] = trace
            if out:
                return out

    # Fallback: detect language-like keys at top level.
    reserved = {"id", "qid", "question_id", "question", "prompt", "query"}
    fallback: Dict[str, str] = {}
    for key, value in question_obj.items():
        if key in reserved:
            continue
        if isinstance(value, (str, list, Mapping)):
            trace = _trace_from_value(value)
            if trace:
                fallback[str(key)] = trace
    return fallback


def load_questions(json_path: str) -> List[Tuple[str, Dict[str, str]]]:
    with open(json_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    if isinstance(data, Mapping):
        if isinstance(data.get("questions"), list):
            raw_questions = data["questions"]
        else:
            raw_questions = [data]
    elif isinstance(data, list):
        raw_questions = data
    else:
        raise ValueError("JSON root must be a list or an object with a 'questions' list.")

    out: List[Tuple[str, Dict[str, str]]] = []
    for idx, q in enumerate(raw_questions, start=1):
        if not isinstance(q, Mapping):
            continue
        qid = str(q.get("question_id") or q.get("qid") or q.get("id") or f"q{idx}")
        traces = _extract_language_traces(q)
        traces = {k: v for k, v in traces.items() if v is not None and str(v).strip() != ""}
        if traces:
            out.append((qid, traces))
    return out


def _metric_distance(
    ref_trace: str,
    target_trace: str,
    alpha: float,
    beta: float,
    gamma: float,
    lam: float,
    dim: int,
) -> Dict[str, float]:
    result = calculate_trace_distance(
        trace_a=ref_trace,
        trace_b=target_trace,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        lam=lam,
        dim=dim,
    )
    return {m: float(result[m]) for m in METRICS}


def export_metric_csvs(
    questions: Sequence[Tuple[str, Dict[str, str]]],
    output_dir: str,
    output_prefix: str,
    reference_lang: str | None,
    alpha: float,
    beta: float,
    gamma: float,
    lam: float,
    dim: int,
) -> List[Path]:
    langs = sorted({lang for _, traces in questions for lang in traces.keys()})
    output_paths: List[Path] = []

    all_rows: Dict[str, List[Dict[str, object]]] = {m: [] for m in METRICS}

    for qid, traces in questions:
        ref = reference_lang if reference_lang in traces else sorted(traces.keys())[0]
        metric_row: Dict[str, Dict[str, object]] = {
            m: {"question_id": qid, "reference_lang": ref} for m in METRICS
        }

        for lang in langs:
            if lang not in traces:
                for m in METRICS:
                    metric_row[m][lang] = ""
                continue

            if lang == ref:
                scores = {"Dsem": 0.0, "Dstr": 0.0, "Dord": 0.0, "Plen": 0.0, "d": 0.0, "Inv": 1.0}
            else:
                scores = _metric_distance(
                    ref_trace=traces[ref],
                    target_trace=traces[lang],
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    lam=lam,
                    dim=dim,
                )
            for m in METRICS:
                metric_row[m][lang] = scores[m]

        for m in METRICS:
            all_rows[m].append(metric_row[m])

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    header = ["question_id", "reference_lang", *langs]

    for m in METRICS:
        out_path = out_dir / f"{output_prefix}_{m}.csv"
        with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(all_rows[m])
        output_paths.append(out_path)

    return output_paths


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Calculate reasoning-trace distance from two traces or from JSON.")
    p.add_argument("trace_a", nargs="?", type=str, help="First trace string; supports lines like '1. ...'")
    p.add_argument("trace_b", nargs="?", type=str, help="Second trace string; supports lines like '1. ...'")

    p.add_argument("--input-json", type=str, help="Input JSON path for batch mode")
    p.add_argument("--output-dir", type=str, default=".", help="Output directory for CSV files")
    p.add_argument("--output-prefix", type=str, default="distance", help="Prefix of output CSV files")
    p.add_argument("--reference-lang", type=str, default=None, help="Reference language code for per-question comparison")

    p.add_argument("--alpha", type=float, default=0.4, help="Weight for Dsem")
    p.add_argument("--beta", type=float, default=0.2, help="Weight for Dstr")
    p.add_argument("--gamma", type=float, default=0.3, help="Weight for Dord")
    p.add_argument("--lam", type=float, default=0.1, help="Weight for Plen")
    p.add_argument("--dim", type=int, default=256, help="Embedding dimension")
    p.add_argument("--pretty", action="store_true", help="Pretty-print JSON output in pair mode")
    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.input_json:
        questions = load_questions(args.input_json)
        if not questions:
            raise ValueError("No valid question/language traces found in input JSON.")

        outputs = export_metric_csvs(
            questions=questions,
            output_dir=args.output_dir,
            output_prefix=args.output_prefix,
            reference_lang=args.reference_lang,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            lam=args.lam,
            dim=args.dim,
        )
        print(json.dumps({"generated_csv": [str(p) for p in outputs]}, ensure_ascii=False, indent=2))
        return

    if args.trace_a is None or args.trace_b is None:
        parser.error("Either provide trace_a trace_b, or use --input-json for batch mode.")

    result = calculate_trace_distance(
        trace_a=args.trace_a,
        trace_b=args.trace_b,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        lam=args.lam,
        dim=args.dim,
    )

    if args.pretty:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
