#!/usr/bin/env python3
"""
Autonomous experiment loop for the macOS/MPS port.

This follows the spirit of program.md:
- keep results.tsv untracked
- commit each train.py experiment
- run train.py with a 10 minute timeout
- keep only improvements, otherwise reset back to the best commit
"""

from __future__ import annotations

import ast
import random
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "train.py"
RESULTS_PATH = ROOT / "results.tsv"
RUN_LOG_PATH = ROOT / "run.log"
MAX_SEQ_LEN = 2048
HEADER = "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n"

MAC_BLOCK_RE = re.compile(
    r"else:\n(?P<body>(?:    .*\n)+?)\n# ---------------------------------------------------------------------------\n# Setup:",
    re.MULTILINE,
)
METRIC_PATTERNS = {
    "val_bpb": re.compile(r"^val_bpb:\s+([0-9.]+)$", re.MULTILINE),
    "peak_vram_mb": re.compile(r"^peak_vram_mb:\s+([0-9.]+)$", re.MULTILINE),
}

SEARCH_SPACE = {
    "ASPECT_RATIO": [24, 32, 40, 48],
    "HEAD_DIM": [32, 64],
    "WINDOW_PATTERN": ["L", "SL", "SSL"],
    "TOTAL_BATCH_SIZE": [2**13, 2**14, 2**15, 2**16],
    "EMBEDDING_LR": [0.15, 0.2, 0.25, 0.3, 0.4],
    "UNEMBEDDING_LR": [0.002, 0.004, 0.006],
    "MATRIX_LR": [0.005, 0.01, 0.015, 0.02, 0.03],
    "SCALAR_LR": [0.1, 0.15, 0.2, 0.25, 0.35],
    "WEIGHT_DECAY": [0.0, 0.05, 0.1, 0.15],
    "DEPTH": [2, 3, 4, 5, 6],
    "DEVICE_BATCH_SIZE": [1, 2, 4, 8],
}
MUTABLE_KEYS = list(SEARCH_SPACE)


def run(cmd: list[str], check: bool = True, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=ROOT,
        check=check,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def run_logged(timeout_seconds: int = 600) -> tuple[bool, dict[str, float]]:
    with RUN_LOG_PATH.open("w") as handle:
        try:
            proc = subprocess.run(
                ["uv", "run", "train.py"],
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds,
                check=False,
                text=True,
            )
        except subprocess.TimeoutExpired:
            return False, {}
    metrics = parse_metrics()
    return proc.returncode == 0 and bool(metrics), metrics


def parse_metrics() -> dict[str, float]:
    text = RUN_LOG_PATH.read_text(errors="replace") if RUN_LOG_PATH.exists() else ""
    metrics = {}
    for key, pattern in METRIC_PATTERNS.items():
        match = pattern.search(text)
        if match:
            metrics[key] = float(match.group(1))
    return metrics


def tail_run_log(lines: int = 40) -> str:
    if not RUN_LOG_PATH.exists():
        return ""
    text = RUN_LOG_PATH.read_text(errors="replace")
    return "\n".join(text.splitlines()[-lines:])


def ensure_results_file() -> None:
    if not RESULTS_PATH.exists():
        RESULTS_PATH.write_text(HEADER)
        return
    text = RESULTS_PATH.read_text()
    if not text.startswith(HEADER):
        RESULTS_PATH.write_text(HEADER)


def read_results() -> list[dict[str, str]]:
    ensure_results_file()
    rows = []
    lines = RESULTS_PATH.read_text().splitlines()
    for line in lines[1:]:
        if not line.strip():
            continue
        commit, val_bpb, memory_gb, status, description = line.split("\t", 4)
        rows.append(
            {
                "commit": commit,
                "val_bpb": val_bpb,
                "memory_gb": memory_gb,
                "status": status,
                "description": description,
            }
        )
    return rows


def append_result(commit: str, val_bpb: float, peak_vram_mb: float, status: str, description: str) -> None:
    memory_gb = peak_vram_mb / 1024 if peak_vram_mb else 0.0
    row = f"{commit}\t{val_bpb:.6f}\t{memory_gb:.1f}\t{status}\t{description}\n"
    with RESULTS_PATH.open("a") as handle:
        handle.write(row)


def best_result() -> tuple[str, float] | None:
    keeps = [row for row in read_results() if row["status"] == "keep"]
    if not keeps:
        return None
    best = keeps[-1]
    return best["commit"], float(best["val_bpb"])


def head_short_sha() -> str:
    return run(["git", "rev-parse", "--short", "HEAD"]).stdout.strip()


def ensure_clean_train_state() -> None:
    status = run(["git", "status", "--short", "--", "train.py"], check=True).stdout.strip()
    if status:
        raise RuntimeError(f"train.py is dirty before starting loop:\n{status}")


def read_mac_settings() -> dict[str, object]:
    text = TRAIN_PATH.read_text()
    match = MAC_BLOCK_RE.search(text)
    if match is None:
        raise RuntimeError("Could not find the non-CUDA hyperparameter block in train.py")
    body = match.group("body")
    settings = {}
    for key in MUTABLE_KEYS:
        assign = re.search(rf"^    {key}\s*=\s*(.+)$", body, re.MULTILINE)
        if assign is None:
            raise RuntimeError(f"Missing assignment for {key} in train.py")
        settings[key] = ast.literal_eval(assign.group(1).strip())
    return settings


def render_value(value: object) -> str:
    if isinstance(value, str):
        return f"\"{value}\""
    return repr(value)


def write_mac_settings(settings: dict[str, object]) -> None:
    text = TRAIN_PATH.read_text()
    match = MAC_BLOCK_RE.search(text)
    if match is None:
        raise RuntimeError("Could not find the non-CUDA hyperparameter block in train.py")
    body = match.group("body")
    new_body = body
    for key, value in settings.items():
        pattern = re.compile(rf"(^    {key}\s*=\s*).*$", re.MULTILINE)
        new_body = pattern.sub(lambda m: m.group(1) + render_value(value), new_body, count=1)
    TRAIN_PATH.write_text(text[: match.start("body")] + new_body + text[match.end("body") :])


def normalize_settings(settings: dict[str, object]) -> dict[str, object]:
    device_batch = int(settings["DEVICE_BATCH_SIZE"])
    min_tokens = device_batch * MAX_SEQ_LEN
    allowed_totals = [min_tokens * factor for factor in (1, 2, 4, 8, 16)]
    total_batch = int(settings["TOTAL_BATCH_SIZE"])
    if total_batch not in allowed_totals:
        settings["TOTAL_BATCH_SIZE"] = min(allowed_totals, key=lambda value: abs(value - total_batch))
    return settings


def mutate_settings(current: dict[str, object], rng: random.Random) -> tuple[dict[str, object], str]:
    candidate = dict(current)
    num_mutations = rng.randint(1, 3)
    mutated_keys = rng.sample(MUTABLE_KEYS, num_mutations)
    for key in mutated_keys:
        choices = [value for value in SEARCH_SPACE[key] if value != candidate[key]]
        candidate[key] = rng.choice(choices)
    candidate = normalize_settings(candidate)
    description = ", ".join(f"{key}={candidate[key]}" for key in mutated_keys)
    return candidate, description


def git_commit(message: str) -> str:
    run(["git", "add", "train.py"])
    run(["git", "commit", "-m", message])
    return head_short_sha()


def git_reset_to(commit: str) -> None:
    run(["git", "reset", "--hard", commit])


def record_baseline() -> tuple[str, float]:
    existing = best_result()
    if existing is not None:
        return existing
    commit = head_short_sha()
    print(f"[baseline] running commit {commit}", flush=True)
    ok, metrics = run_logged()
    if not ok:
        print("[baseline] failed", flush=True)
        print(tail_run_log(), flush=True)
        append_result(commit, 0.0, 0.0, "crash", "baseline")
        raise RuntimeError("Baseline failed")
    val_bpb = metrics["val_bpb"]
    peak_vram_mb = metrics.get("peak_vram_mb", 0.0)
    append_result(commit, val_bpb, peak_vram_mb, "keep", "baseline")
    print(f"[baseline] keep {commit} val_bpb={val_bpb:.6f}", flush=True)
    return commit, val_bpb


def main() -> int:
    ensure_results_file()
    ensure_clean_train_state()

    best_commit, best_bpb = record_baseline()
    rng = random.Random(time.time_ns())
    iteration = 0

    while True:
        iteration += 1
        current = read_mac_settings()
        candidate, description = mutate_settings(current, rng)
        write_mac_settings(candidate)

        commit_message = f"exp: {description}"
        exp_commit = git_commit(commit_message[:120])
        print(f"[exp {iteration}] {exp_commit} {description}", flush=True)

        ok, metrics = run_logged()
        if not ok:
            print(f"[exp {iteration}] crash", flush=True)
            print(tail_run_log(), flush=True)
            append_result(exp_commit, 0.0, 0.0, "crash", description)
            git_reset_to(best_commit)
            continue

        val_bpb = metrics["val_bpb"]
        peak_vram_mb = metrics.get("peak_vram_mb", 0.0)
        if val_bpb < best_bpb:
            append_result(exp_commit, val_bpb, peak_vram_mb, "keep", description)
            best_commit = exp_commit
            best_bpb = val_bpb
            print(f"[exp {iteration}] keep {exp_commit} val_bpb={val_bpb:.6f}", flush=True)
        else:
            append_result(exp_commit, val_bpb, peak_vram_mb, "discard", description)
            print(f"[exp {iteration}] discard {exp_commit} val_bpb={val_bpb:.6f} best={best_bpb:.6f}", flush=True)
            git_reset_to(best_commit)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise
