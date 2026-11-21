#!/usr/bin/env python3
"""Interactive Marathi -> Roman transliteration in the terminal.

Just type any Marathi word or sentence and press Enter.
Type 'quit' or 'exit' or press Ctrl+C to stop.

Run:
    .\.venv\Scripts\python.exe -u .\translit_terminal.py
"""
import sys
import os
import re
import math
import time

# ── fix Windows terminal encoding for Devanagari ─────────────────────────────
# Windows default code page 437 / 1252 has no Devanagari.
# We switch the console to UTF-8 (cp 65001) programmatically so the user
# does not need to run `chcp 65001` manually every session.
if sys.platform == "win32":
    import ctypes
    # set console input + output code page to UTF-8
    ctypes.windll.kernel32.SetConsoleOutputCP(65001)
    ctypes.windll.kernel32.SetConsoleCP(65001)

# reconfigure Python's own stdin/stdout/stderr to UTF-8
for _s in (sys.stdin, sys.stdout, sys.stderr):
    if _s is not None and hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# ── load the in-process engine (once) ────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "app", "ai4bharat", "transliteration", "transformer"))

from custom_interactive import Transliterator  # noqa

import logging
logging.getLogger("fairseq").setLevel(logging.WARNING)
logging.getLogger("root").setLevel(logging.WARNING)

BEAM      = 6
NBEST     = 5
# ── ranking strategy ─────────────────────────────────────────────────────────
# Beam search scores are log2-probabilities (always <= 0).
# 'max' : pick the HIGHEST score candidate (closest to 0) = most confident.
#         This is the standard beam-search result.
# 'min' : pick the LOWEST  score candidate (most negative) = least confident.
#         Selects the word the model considers the hardest / least probable.
RANK_BY   = "min"    # <── change to "max" for standard beam-search behaviour
DATA_BIN  = os.path.join(ROOT, "models", "indic-en", "corpus-bin")
CKPT      = os.path.join(ROOT, "models", "indic-en", "transformer", "indicxlit.pt")
LANG_LIST = os.path.join(ROOT, "inference", "cli", "indic-en", "lang_list.txt")
MR_WORD   = re.compile(r"[\u0900-\u097F]+")
H_RE      = re.compile(r"^H-(\d+)\t(-?[\d.]+)\t(.*)$")
DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

print("Loading model ...", flush=True)
t0 = time.time()
engine = Transliterator(
    data_bin_dir=DATA_BIN,
    model_checkpoint_path=CKPT,
    lang_pairs_csv="mr-en",
    lang_list_file=LANG_LIST,
    beam=BEAM,
    batch_size=32,
)
on_cuda = False
try:
    on_cuda = next(engine.models[0].parameters()).is_cuda
except Exception:
    pass
print(f"Ready in {time.time()-t0:.1f}s  (device: {'cuda' if on_cuda else 'cpu'})", flush=True)
print(f"Ranking : {RANK_BY.upper()}  "
      f"({'most confident' if RANK_BY == 'max' else 'least confident — lowest score in beam'})",
      flush=True)
print("─" * 50, flush=True)
print("Type Marathi text and press Enter.", flush=True)
print("Type 'quit' to exit.\n", flush=True)


def select_final(candidates, rank):
    """Pick the final answer from (candidate, prob_pct) pairs.

    rank='max' -> highest prob (most confident) = standard beam result
    rank='min' -> lowest  prob (least confident) = smallest value in beam
    """
    if not candidates:
        return None, None
    if rank == "min":
        return min(candidates, key=lambda c: c[1])   # lowest probability %
    return max(candidates, key=lambda c: c[1])        # highest probability %


def transliterate(text):
    text = text.translate(DEV_DIGITS).replace("।", ".").replace("॥", ".")
    words = MR_WORD.findall(text)
    if not words:
        return None, []

    proc = ["__mr__ " + " ".join(list(w.lower())) for w in words]
    raw  = engine.translate(proc, nbest=NBEST)

    # parse hypotheses: {word_idx: [(candidate, prob_pct), ...]}
    # Fairseq returns them sorted highest-score-first (rank 1 = most confident).
    hyps = {}
    for line in raw.splitlines():
        m = H_RE.match(line)
        if m:
            idx   = int(m.group(1))
            score = float(m.group(2))          # log2-probability (always <= 0)
            cand  = m.group(3).replace(" ", "")
            prob  = round(2.0 ** score * 100, 1)   # convert to percentage
            hyps.setdefault(idx, []).append((cand, prob))

    # rebuild sentence: replace each Marathi word with the chosen final answer
    result = text
    details = []
    for i, w in enumerate(words):
        cands = hyps.get(i, [])
        final_cand, final_prob = select_final(cands, RANK_BY)
        chosen = final_cand if final_cand else w
        result = result.replace(w, chosen, 1)
        details.append((w, cands, chosen, final_prob))

    return result, details


# ── main loop ────────────────────────────────────────────────────────────────
while True:
    try:
        line = input("मराठी > ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye!", flush=True)
        break

    if not line:
        continue
    if line.lower() in ("quit", "exit", "q"):
        print("Goodbye!", flush=True)
        break

    t0 = time.time()
    result, details = transliterate(line)
    elapsed = time.time() - t0

    if result is None:
        print("  (no Marathi characters found)\n", flush=True)
        continue

    # ── top-1 result ─────────────────────────────────────────────────────────
    strategy = "LOWEST score (least confident)" if RANK_BY == "min" else "HIGHEST score (most confident)"
    print(f"\n  Final [{RANK_BY}] : {result}  ← {strategy}", flush=True)
    print(f"  Time           : {elapsed:.2f}s\n", flush=True)

    # ── per-word candidates ───────────────────────────────────────────────────
    if details:
        print(f"  {'Marathi':<18} {'All candidates':<44} {'➜ chosen'}", flush=True)
        print("  " + "─" * 72, flush=True)
        for mr_word, cands, chosen, chosen_prob in details:
            cand_str = "  ".join(f"{c}({p}%)" for c, p in cands)
            marker   = f"➜ {chosen} ({chosen_prob}%)"
            print(f"  {mr_word:<18} {cand_str:<44}  {marker}", flush=True)
    print(flush=True)
