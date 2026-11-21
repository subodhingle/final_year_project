#!/usr/bin/env python3
"""Run IndicXlit's Fairseq beam decoder for Marathi transliteration.

Candidate ranking strategy (--rank):
  max  (default) : pick the candidate with the HIGHEST score (closest to 0).
                   This is the standard beam-search result — the most probable
                   transliteration according to the model.
  min            : pick the candidate with the LOWEST score (most negative).
                   This is the LEAST confident candidate in the beam — useful
                   for exploring what the model considers the worst option, or
                   as a deliberate alternative-selection strategy.

Both strategies show all nbest candidates ranked 1..N; only the "final answer"
line changes.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LANG_PAIRS = {
    "roman-to-marathi": "en-as,en-bn,en-brx,en-gom,en-gu,en-hi,en-kn,en-ks,en-mai,en-ml,en-mni,en-mr,en-ne,en-or,en-pa,en-sa,en-sd,en-si,en-ta,en-te,en-ur",
    "marathi-to-roman": "as-en,bn-en,brx-en,gom-en,gu-en,hi-en,kn-en,ks-en,mai-en,ml-en,mni-en,mr-en,ne-en,or-en,pa-en,sa-en,sd-en,si-en,ta-en,te-en,ur-en",
}


def select_final(candidates: list[tuple[float, str]], rank: str) -> tuple[float, str] | None:
    """Return the final answer from a list of (score, word) candidates.

    rank='max'  -> highest log-prob score  (most confident, standard beam result)
    rank='min'  -> lowest  log-prob score  (least confident candidate in beam)

    Scores are log2-probabilities (always <= 0). Fairseq already returns them
    sorted highest-first, so max = candidates[0] and min = candidates[-1].
    """
    if not candidates:
        return None
    if rank == "min":
        return min(candidates, key=lambda c: c[0])   # most negative = least confident
    return max(candidates, key=lambda c: c[0])        # closest to 0  = most confident


def run(text: str, direction: str, beam: int, nbest: int) -> list[tuple[str, list[tuple[float, str]]]]:
    if beam < 1 or nbest < 1 or nbest > beam:
        raise ValueError("beam and nbest must be positive, with nbest <= beam")

    reverse = direction == "marathi-to-roman"
    folder      = ROOT / "inference" / "cli" / ("indic-en" if reverse else "en-indic")
    model_dir   = ROOT / "models"    / ("indic-en" if reverse else "en-indic")
    checkpoint  = model_dir / "transformer" / "indicxlit.pt"
    data_bin    = model_dir / "corpus-bin"
    if not checkpoint.is_file() or not (data_bin / "dict.en.txt").is_file():
        raise FileNotFoundError("IndicXlit checkpoint or Fairseq dictionaries are missing under models/")

    words = text.split()
    if not words:
        return []

    # IndicXlit is character-level: each word is space-joined characters.
    prepared = "\n".join(" ".join(word) for word in words) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as src:
        src.write(prepared)
        input_path = Path(src.name)

    source_lang, target_lang = ("mr", "en") if reverse else ("en", "mr")
    fairseq_cli = Path(sys.executable).parent / "fairseq-interactive.exe"
    executable  = str(fairseq_cli) if fairseq_cli.is_file() else "fairseq-interactive"
    command = [
        executable, str(data_bin), "--path", str(checkpoint),
        "--task", "translation_multi_simple_epoch",
        "--beam", str(beam), "--nbest", str(nbest),
        "--source-lang", source_lang, "--target-lang", target_lang,
        # indic->en uses --encoder-langtok src; en->indic uses tgt.
        "--encoder-langtok", "src" if reverse else "tgt",
        "--lang-dict", "lang_list.txt", "--lang-pairs", LANG_PAIRS[direction],
        "--input", str(input_path),
    ]
    try:
        child_env = os.environ.copy()
        child_env["PYTHONIOENCODING"] = "utf-8"
        completed = subprocess.run(
            command, cwd=folder, capture_output=True,
            text=True, encoding="utf-8", env=child_env,
        )
    finally:
        input_path.unlink(missing_ok=True)

    if completed.returncode:
        raise RuntimeError(
            completed.stderr.strip() or completed.stdout.strip() or "Fairseq inference failed"
        )

    # Parse H- hypothesis lines -> {word_index: [(score, candidate), ...]}
    # Fairseq already returns them sorted highest-score-first.
    hypotheses: dict[int, list[tuple[float, str]]] = {}
    for line in completed.stdout.splitlines():
        m = re.match(r"H-(\d+)\t(-?[\d.]+)\t(.*)$", line)
        if m:
            idx, score, cand = m.groups()
            hypotheses.setdefault(int(idx), []).append((float(score), cand.replace(" ", "")))

    return [(word, hypotheses.get(i, [])) for i, word in enumerate(words)]


def main() -> int:
    # UTF-8 console so Devanagari prints correctly on Windows.
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        ctypes.windll.kernel32.SetConsoleCP(65001)
    for _s in (sys.stdout, sys.stderr):
        if hasattr(_s, "reconfigure"):
            try:
                _s.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("text",
                        help="one or more whitespace-separated Marathi (or Roman) words")
    parser.add_argument("--direction", choices=LANG_PAIRS,
                        default="marathi-to-roman")
    parser.add_argument("--beam",  type=int, default=6,
                        help="beam width  (default 6)")
    parser.add_argument("--nbest", type=int, default=5,
                        help="candidates to show, must be <= beam  (default 5)")
    parser.add_argument("--rank",  choices=["max", "min"], default="max",
                        help="how to pick the FINAL answer from the beam:\n"
                             "  max (default) = highest score = most confident\n"
                             "  min           = lowest  score = least confident")
    args = parser.parse_args()

    try:
        results = run(args.text, args.direction, args.beam, args.nbest)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    sep = "─" * 52
    for word, candidates in results:
        print(f"\n{word}")
        print(sep)
        if not candidates:
            print("  (no hypotheses returned)")
            continue

        # ── all beam candidates ───────────────────────────────────────────────
        for rank, (score, cand) in enumerate(candidates, 1):
            prob = 2.0 ** score * 100
            print(f"  {rank}. {cand:<22}  score {score:+.4f}  ({prob:.1f}%)")

        # ── final answer picked by the chosen ranking strategy ────────────────
        final_score, final_cand = select_final(candidates, args.rank)
        strategy_label = (
            "HIGHEST score (most confident)"  if args.rank == "max"
            else "LOWEST score  (least confident)"
        )
        print(f"\n  ➜  Final answer [{args.rank}] : {final_cand}"
              f"  [{strategy_label},  score {final_score:+.4f}]")
        print(sep)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
