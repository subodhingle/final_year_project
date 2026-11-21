#!/usr/bin/env python3
"""Evaluate the finetuned checkpoint on the test set.

Runs fairseq-generate on corpus-bin-mr test split, then computes:
  - Top-1 Exact Match Accuracy
  - Character Error Rate (CER)
  - Word Error Rate (WER)
  - Sample predictions (first 20)

Compares finetuned checkpoint vs original indicxlit.pt on the same test set.
"""
import re
import sys
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
DATA_BIN = str(ROOT / "corpus-bin-mr")
LANG_LIST = str(ROOT / "inference" / "cli" / "indic-en" / "lang_list.txt")

CHECKPOINTS = {
    "original":   str(ROOT / "models" / "indic-en" / "transformer" / "indicxlit_original.pt"),
    "finetuned":  str(ROOT / "transformer" / "finetune_mr" / "checkpoint_best.pt"),
}

def configure_utf8_output():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

def run_generate(ckpt_path, label):
    cmd = [
        VENV_PY, "-m", "fairseq_cli.generate", DATA_BIN,
        "--path", ckpt_path,
        "--task", "translation_multi_simple_epoch",
        "--gen-subset", "test",
        "--beam", "5", "--nbest", "1",
        "--source-lang", "mr", "--target-lang", "en",
        "--encoder-langtok", "src",
        "--lang-dict", LANG_LIST,
        "--lang-pairs", "mr-en",
        "--max-tokens", "8192",
        "--num-workers", "0",
    ]
    env = os.environ.copy(); env["PYTHONIOENCODING"] = "utf-8"
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", env=env)
    elapsed = time.time() - t0
    if p.returncode != 0:
        print(f"[{label}] generate FAILED:\n{p.stderr[-800:]}", flush=True)
        return None, elapsed
    return p.stdout, elapsed

def cer(ref, hyp):
    """Character Error Rate via simple edit distance."""
    r, h = list(ref), list(hyp)
    d = [[0]*(len(h)+1) for _ in range(len(r)+1)]
    for i in range(len(r)+1): d[i][0] = i
    for j in range(len(h)+1): d[0][j] = j
    for i in range(1,len(r)+1):
        for j in range(1,len(h)+1):
            d[i][j] = d[i-1][j-1] if r[i-1]==h[j-1] else 1+min(d[i-1][j],d[i][j-1],d[i-1][j-1])
    return d[len(r)][len(h)] / max(len(r), 1)

def parse_output(stdout):
    """Parse fairseq generate output -> {id: (src_chars, hyp_word)}"""
    results = {}
    for line in stdout.splitlines():
        # S- lines: source chars (original Marathi char sequence)
        ms = re.match(r"^S-(\d+)\t(.+)$", line)
        if ms:
            idx = int(ms.group(1))
            results.setdefault(idx, {})["src"] = ms.group(2).strip()
        # T- lines: reference (ground truth)
        mt = re.match(r"^T-(\d+)\t(.+)$", line)
        if mt:
            idx = int(mt.group(1))
            results.setdefault(idx, {})["ref"] = mt.group(2).replace(" ", "").strip()
        # H- lines: hypothesis (top-1 prediction)
        mh = re.match(r"^H-(\d+)\t\S+\t(.+)$", line)
        if mh:
            idx = int(mh.group(1))
            results.setdefault(idx, {})["hyp"] = mh.group(2).replace(" ", "").strip()
    return results

def score(results):
    """Return (exact_match_pct, avg_cer, avg_wer)."""
    exact = cer_sum = wer_sum = n = 0
    for idx in sorted(results):
        r = results[idx]
        ref = r.get("ref", "")
        hyp = r.get("hyp", "")
        if not ref: continue
        n += 1
        exact += (hyp == ref)
        cer_sum += cer(ref, hyp)
        wer_sum += (0 if hyp == ref else 1)   # WER for single words = 0 or 1
    if n == 0: return 0, 0, 0
    return 100*exact/n, 100*cer_sum/n, 100*wer_sum/n

def main():
    configure_utf8_output()
    required_paths = [DATA_BIN, LANG_LIST, VENV_PY] + list(CHECKPOINTS.values())
    missing = [path for path in required_paths if not Path(path).exists()]
    if missing:
        print("[setup] missing required paths:", flush=True)
        for path in missing:
            print(f"  - {path}", flush=True)
        return 1

    all_results = {}
    had_failure = False
    for label, ckpt in CHECKPOINTS.items():
        print(f"\n[{label}] running fairseq-generate on test set ...", flush=True)
        stdout, elapsed = run_generate(ckpt, label)
        if stdout is None:
            had_failure = True
            continue
        results = parse_output(stdout)
        em, avg_cer, avg_wer = score(results)
        all_results[label] = results
        print(f"[{label}]  test pairs={len(results)}  elapsed={elapsed:.1f}s", flush=True)
        print(f"[{label}]  Top-1 Exact Match : {em:.2f}%", flush=True)
        print(f"[{label}]  Avg CER           : {avg_cer:.2f}%", flush=True)
        print(f"[{label}]  Avg WER           : {avg_wer:.2f}%", flush=True)

    # Comparison table
    if len(all_results) == 2:
        orig = all_results["original"]; fine = all_results["finetuned"]
        em_o, cer_o, wer_o = score(orig)
        em_f, cer_f, wer_f = score(fine)
        print("\n" + "="*60, flush=True)
        print(f"{'Metric':<28}{'Original':>14}{'Finetuned':>14}", flush=True)
        print("-"*60, flush=True)
        print(f"{'Top-1 Exact Match (%)':<28}{em_o:>14.2f}{em_f:>14.2f}", flush=True)
        print(f"{'Avg CER (%)':<28}{cer_o:>14.2f}{cer_f:>14.2f}", flush=True)
        print(f"{'Avg WER (%)':<28}{wer_o:>14.2f}{wer_f:>14.2f}", flush=True)
        print("="*60, flush=True)

    # Sample predictions from the finetuned model
    if "finetuned" in all_results:
        fine = all_results["finetuned"]
        ids = sorted(fine.keys())[:20]
        print(f"\n=== Sample predictions (finetuned, first 20) ===", flush=True)
        print(f"{'Marathi':<18}{'Reference':<18}{'Predicted':<18}{'CER%':>6}{'Match':>6}", flush=True)
        print("-"*68, flush=True)
        for idx in ids:
            r = fine[idx]
            src = r.get("src","").replace("__mr__","").replace(" ","")
            ref = r.get("ref","")
            hyp = r.get("hyp","")
            c = cer(ref, hyp)*100
            m = "Y" if hyp==ref else "N"
            print(f"{src:<18}{ref:<18}{hyp:<18}{c:>6.1f}{m:>6}", flush=True)

    if had_failure:
        return 1
    if len(all_results) != len(CHECKPOINTS):
        print("[setup] evaluation did not produce results for both checkpoints.", flush=True)
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
