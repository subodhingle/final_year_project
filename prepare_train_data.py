#!/usr/bin/env python3
"""Build char-level Marathi->English training data from subodh_trial_csv.csv.

Output (under corpus/): train_mr.mr/.en, valid_mr.mr/.en, test_mr.mr/.en
Each line = one word, characters space-separated:
   Marathi source : "र ा व े र"
   English target : "r a v e r"   (lowercased)

Cleaning rules (keeps only faithful romanizations, drops abbreviations):
  - Marathi token must be purely Devanagari (U+0900-U+097F).
  - English token must be purely ASCII alphabetic (drops "Bk.", "Kh.", digits).
  - Marathi and English must have the SAME number of whitespace tokens, and
    every token pair must pass the two rules above.
Split is by villageCode (all pairs of one village stay in one split) to avoid
data leakage. A fixed seed makes it reproducible.
"""
import csv
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "subodh_trial_csv.csv"
OUT = ROOT / "corpus"
OUT.mkdir(exist_ok=True)

MR_IDX = 3   # villageNameLocal  (Devanagari)
EN_IDX = 2   # villageNameEnglish (Roman)
ID_IDX = 1   # villageCode

MR_RE = re.compile(r"^[\u0900-\u097F]+$")
EN_RE = re.compile(r"^[A-Za-z]+$")
SEED = 42


def char_split(word):
    return " ".join(list(word))


def main():
    by_village = defaultdict(list)  # villageCode -> list of (mr_word, en_word)
    total_rows = 0
    dropped_abbrev = 0
    dropped_mismatch = 0
    with open(CSV_PATH, encoding="utf-8-sig") as f:
        r = csv.reader(f)
        next(r, None)  # header
        for row in r:
            if len(row) <= max(MR_IDX, EN_IDX, ID_IDX):
                continue
            total_rows += 1
            mr = row[MR_IDX].strip()
            en = row[EN_IDX].strip()
            vid = row[ID_IDX].strip() or str(total_rows)
            if not mr or not en:
                continue
            mr_toks = mr.split()
            en_toks = en.split()
            if len(mr_toks) != len(en_toks):
                dropped_mismatch += 1
                continue
            ok = True
            for m, e in zip(mr_toks, en_toks):
                if not MR_RE.match(m) or not EN_RE.match(e):
                    ok = False
                    break
            if not ok:
                dropped_abbrev += 1
                continue
            for m, e in zip(mr_toks, en_toks):
                by_village[vid].append((m, e.lower()))

    # split by village -> 80/10/10
    villages = sorted(by_village.keys())
    random.seed(SEED)
    random.shuffle(villages)
    n = len(villages)
    n_train = int(n * 0.8)
    n_valid = int(n * 0.1)
    splits = {
        "train": villages[:n_train],
        "valid": villages[n_train:n_train + n_valid],
        "test": villages[n_train + n_valid:],
    }

    counts = {}
    for split, vids in splits.items():
        pairs = []
        for v in vids:
            pairs.extend(by_village[v])
        random.seed(SEED)
        random.shuffle(pairs)
        counts[split] = len(pairs)
        mr_path = OUT / f"{split}_mr.mr"
        en_path = OUT / f"{split}_mr.en"
        with open(mr_path, "w", encoding="utf-8") as fm, \
             open(en_path, "w", encoding="utf-8") as fe:
            for m, e in pairs:
                fm.write(char_split(m) + "\n")
                fe.write(char_split(e) + "\n")

    print(f"CSV rows seen           : {total_rows}")
    print(f"dropped (token mismatch): {dropped_mismatch}")
    print(f"dropped (abbrev/non-alpha): {dropped_abbrev}")
    print(f"villages                : {n}  (train {len(splits['train'])}, "
          f"valid {len(splits['valid'])}, test {len(splits['test'])})")
    print(f"clean word pairs -> train: {counts['train']}, "
          f"valid: {counts['valid']}, test: {counts['test']}")
    print(f"total clean pairs       : {sum(counts.values())}")
    print("\nsamples (marathi -> english):")
    sample = []
    with open(OUT / "train_mr.mr", encoding="utf-8") as fm, \
         open(OUT / "train_mr.en", encoding="utf-8") as fe:
        for m, e in zip(fm, fe):
            sample.append((m.strip().replace(" ", ""), e.strip().replace(" ", "")))
            if len(sample) >= 6:
                break
    for m, e in sample:
        print(f"   {m} -> {e}")


if __name__ == "__main__":
    main()
