#!/usr/bin/env python3
"""
Marathi -> Roman transliteration web page.

A tiny zero-dependency local web page (Python stdlib http.server only) that
loads the trained IndicXlit Marathi->English transformer ONCE into memory and
serves beam-search transliteration. Paste Marathi (Devanagari) text, get the
Roman transliteration.

Run with the project venv (which has torch CUDA + fairseq):
    e:\\indicxlict\\.venv\\Scripts\\python.exe -u e:\\indicxlict\\translit_page.py
Then open http://127.0.0.1:8080 in a browser.

Notes
-----
* The model is the IndicXlit multilingual transformer; we restrict it to the
  Marathi pair ("mr-en") and prepend the "__mr__" source-language token, which is
  exactly what IndicXlit's `indic-en` CLI does with `--encoder-langtok src`.
* Beam width is set when the model is loaded (fairseq builds the generator with a
  fixed beam). The first request for a given beam width loads it (~2s on GPU);
  later requests reuse the cached model. Top-N (nbest) is chosen per request.
* nvidia-smi cannot see GPU processes on this Windows WDDM box, but the model is
  verified on CUDA: `next(model.parameters()).is_cuda == True`.
"""

import json
import logging
import math
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# --- Make the in-process IndicXlit Transliterator importable -----------------
TRANSFORMER_DIR = r"e:\indicxlict\app\ai4bharat\transliteration\transformer"
if TRANSFORMER_DIR not in sys.path:
    sys.path.insert(0, TRANSFORMER_DIR)
from custom_interactive import Transliterator  # noqa: E402

# --- Paths to YOUR trained indic-en model artifacts ---------------------------
DATA_BIN = r"e:\indicxlict\models\indic-en\corpus-bin"
CHECKPOINT = r"e:\indicxlict\models\indic-en\transformer\indicxlit.pt"
LANG_LIST = r"e:\indicxlict\inference\cli\indic-en\lang_list.txt"
LANG_PAIRS = "mr-en"          # Marathi only
DEFAULT_BEAM = 6
MAX_BEAM = 10
PORT = 8080

# Devanagari (Marathi script) word runs. U+0900..U+097F covers Marathi.
MR_WORD = re.compile(r"[\u0900-\u097F]+")
H_RE = re.compile(r"^H-(\d+)\t(-?[\d.]+)\t(.*)$")
# Native numerals and the danda full-stop are passed through (not transliterated),
# matching IndicXlit's behavior.
DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

# Quiet fairseq's chatty INFO logging on the console.
logging.getLogger("fairseq").setLevel(logging.WARNING)


def _maybe_save_history(input_text, output_text, beam, nbest, device, elapsed):
    """Optionally log this transliteration to PostgreSQL. Never breaks the page.

    Enabled only when db_config.json has "save_history": true. Otherwise no DB
    connection is made. Failures are logged and swallowed.
    """
    try:
        import db
        return bool(db.save_history(input_text, output_text, beam, nbest, device, elapsed))
    except Exception as e:
        print(f"[db] history save skipped: {e}")
        return False

# --- Engine cache: one Transliterator per beam width, built lazily ------------
_engines: dict[int, "Transliterator"] = {}
_lock = threading.Lock()


def get_engine(beam: int) -> "Transliterator":
    """Return a cached Transliterator for the requested beam, building if needed."""
    beam = max(1, min(int(beam), MAX_BEAM))
    with _lock:
        eng = _engines.get(beam)
        if eng is None:
            print(f"[engine] loading model for beam={beam} ...", flush=True)
            t0 = time.time()
            eng = Transliterator(
                data_bin_dir=DATA_BIN,
                model_checkpoint_path=CHECKPOINT,
                lang_pairs_csv=LANG_PAIRS,
                lang_list_file=LANG_LIST,
                beam=beam,
                batch_size=32,
            )
            on_cuda = False
            try:
                on_cuda = next(eng.models[0].parameters()).is_cuda
            except Exception:
                pass
            print(f"[engine] ready beam={beam} in {time.time()-t0:.1f}s  (cuda={on_cuda})",
                  flush=True)
            _engines[beam] = eng
        return eng


def parse_hypotheses(translation_str: str):
    """Parse fairseq H- lines into {id: [(word, score), ...]} (score is log2-prob)."""
    hyps: dict[int, list[tuple[str, float]]] = {}
    for line in translation_str.splitlines():
        m = H_RE.match(line)
        if m:
            idx = int(m.group(1))
            score = float(m.group(2))
            word = m.group(3).replace(" ", "")
            hyps.setdefault(idx, []).append((word, score))
    return hyps


def transliterate_text(text: str, beam: int, nbest: int):
    """Transliterate Marathi text to Roman, preserving lines and non-Devanagari chars.

    Returns (full_transliteration_str, per_word_details, elapsed_seconds).
    """
    eng = get_engine(beam)
    nbest = max(1, min(int(nbest), beam))
    t0 = time.time()

    out_lines = []
    details = []  # list per input line of list of {src, cands:[{w, prob}]}

    with _lock:
        for raw_line in text.splitlines():
            # Pass native digits/danda through unchanged; keep Devanagari words only.
            line = raw_line.translate(DEV_DIGITS).replace("।", ".").replace("॥", ".")
            words = MR_WORD.findall(line)
            if not words:
                out_lines.append(line)
                details.append([])
                continue
            # IndicXlit char-level prep: lowercase + space-join chars + __mr__ token
            proc = ["__mr__ " + " ".join(list(w.lower())) for w in words]
            res = eng.translate(proc, nbest=nbest)
            hyps = parse_hypotheses(res)

            # Rebuild the line by replacing each Marathi word with its top-1 roman form.
            rebuilt = line
            line_detail = []
            for i, w in enumerate(words):
                cands = hyps.get(i, [])
                cand_objs = [
                    {"w": cw, "prob": round(2.0 ** sc, 4)} for cw, sc in cands[:nbest]
                ]
                line_detail.append({"src": w, "cands": cand_objs})
                top1 = cands[0][0] if cands else w
                rebuilt = rebuilt.replace(w, top1, 1)
            out_lines.append(rebuilt)
            details.append(line_detail)

    return "\n".join(out_lines), details, time.time() - t0


# --- HTTP server --------------------------------------------------------------
PAGE_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Marathi &rarr; Roman Transliteration (IndicXlit, beam search)</title>
<style>
  :root { color-scheme: light; }
  body { font-family: "Noto Sans Devanagari","Nirmala UI","Mangal",system-ui,Segoe UI,Arial,sans-serif;
         max-width: 880px; margin: 28px auto; padding: 0 18px; color:#1b1b1b; }
  h1 { font-size: 1.35rem; margin: 0 0 4px; }
  .sub { color:#666; font-size: .85rem; margin-bottom: 18px; }
  label { font-size: .8rem; color:#444; }
  textarea, .out { width: 100%; box-sizing: border-box; font-size: 1.05rem;
                   padding: 10px; border:1px solid #ccc; border-radius:8px; }
  textarea { min-height: 110px; font-family: "Noto Sans Devanagari","Nirmala UI","Mangal",monospace; }
  .out { min-height: 70px; background:#f7f9fb; white-space: pre-wrap; }
  .row { display:flex; gap:18px; align-items:flex-end; margin: 14px 0; flex-wrap:wrap; }
  .row > div { display:flex; flex-direction:column; gap:4px; }
  select { padding:6px 8px; border:1px solid #ccc; border-radius:8px; font-size:.95rem; }
  button { padding:9px 18px; border:0; border-radius:8px; background:#2563eb; color:#fff;
           font-size:.95rem; cursor:pointer; }
  button:disabled { background:#9bb4dd; cursor:default; }
  table { border-collapse: collapse; width:100%; margin-top:10px; font-size:.9rem; }
  td, th { border:1px solid #e3e3e3; padding:5px 8px; text-align:left; vertical-align:top; }
  th { background:#f0f3f7; }
  .src { font-family:"Noto Sans Devanagari","Nirmala UI","Mangal",monospace; }
  .prob { color:#888; font-variant-numeric: tabular-nums; }
  .meta { color:#666; font-size:.8rem; margin-top:8px; }
  .err { color:#b00020; }
</style>
</head>
<body>
<h1>Marathi &rarr; Roman Transliteration</h1>
<div class="sub">IndicXlit transformer &middot; Marathi-only (<code>mr&minus;en</code>) &middot; beam search &middot; GPU</div>

<label for="in">Paste Marathi (Devanagari) text:</label>
<textarea id="in" autofocus>आडगाव रावेर अकोले पाटण बीड
नागपूर सातारा परभणी</textarea>

<div class="row">
  <div>
    <label for="beam">Beam width</label>
    <select id="beam">
      <option>1</option><option>2</option><option>3</option>
      <option>4</option><option>5</option><option selected>6</option>
      <option>7</option><option>8</option><option>9</option><option>10</option>
    </select>
  </div>
  <div>
    <label for="nbest">Top-N candidates (per word)</label>
    <select id="nbest"><option selected>1</option></select>
  </div>
  <div>
    <button id="go">Transliterate</button>
  </div>
</div>

<label>Transliteration (top-1 per word):</label>
<div class="out" id="out"></div>
<div id="meta" class="meta"></div>

<div id="detail"></div>

<script>
const $ = id => document.getElementById(id);
function fillNbest() {
  const beam = parseInt($("beam").value, 10);
  const cur = parseInt($("nbest").value, 10) || 1;
  const sel = $("nbest");
  sel.innerHTML = "";
  for (let n = 1; n <= beam; n++) {
    const o = document.createElement("option");
    o.value = n; o.textContent = n;
    if (n === Math.min(cur, beam)) o.selected = true;
    sel.appendChild(o);
  }
}
$("beam").addEventListener("change", fillNbest);
fillNbest();

$("go").addEventListener("click", async () => {
  const text = $("in").value;
  const beam = parseInt($("beam").value, 10);
  const nbest = parseInt($("nbest").value, 10);
  const btn = $("go"); btn.disabled = true; const old = btn.textContent;
  btn.textContent = "Transliterating\u2026";
  $("out").textContent = ""; $("meta").textContent = ""; $("detail").innerHTML = "";
  try {
    const r = await fetch("/transliterate", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({text, beam, nbest})
    });
    const j = await r.json();
    if (j.error) { $("meta").innerHTML = '<span class="err">'+j.error+'</span>'; return; }
    $("out").textContent = j.full;
    $("meta").textContent = `beam=${j.beam}  nbest=${j.nbest}  device=${j.device}  ${j.elapsed.toFixed(2)}s` + (j.db_saved ? "  db:saved" : "");
    if (j.nbest > 1) {
      let h = '<table><tr><th>Marathi</th><th>Candidates (prob)</th></tr>';
      for (const line of j.details) {
        for (const w of line) {
          const cands = w.cands.map(c => `${c.w} <span class="prob">(${(c.prob*100).toFixed(1)}%)</span>`).join(" &middot; ");
          h += `<tr><td class="src">${w.src}</td><td>${cands}</td></tr>`;
        }
      }
      h += '</table>';
      $("detail").innerHTML = h;
    }
  } catch (e) { $("meta").innerHTML = '<span class="err">Request failed: '+e+'</span>'; }
  finally { btn.disabled = false; btn.textContent = old; }
});
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # quieter console
        pass

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            self._send(200, PAGE_HTML.encode("utf-8"), "text/html; charset=utf-8")
        else:
            self._send(404, b"Not found", "text/plain")

    def do_POST(self):
        if self.path != "/transliterate":
            self._send(404, b"Not found", "text/plain")
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n) or b"{}")
            text = str(data.get("text", ""))
            beam = int(data.get("beam", DEFAULT_BEAM))
            nbest = int(data.get("nbest", 1))
        except Exception as e:
            self._send(400, json.dumps({"error": f"bad request: {e}"}).encode(),
                       "application/json")
            return
        try:
            full, details, elapsed = transliterate_text(text, beam, nbest)
            eng = get_engine(beam)
            on_cuda = False
            try:
                on_cuda = next(eng.models[0].parameters()).is_cuda
            except Exception:
                pass
            resp = {
                "full": full,
                "details": details,
                "beam": beam,
                "nbest": max(1, min(nbest, beam)),
                "elapsed": elapsed,
                "device": "cuda" if on_cuda else "cpu",
            }
            resp["db_saved"] = _maybe_save_history(text, full, beam, resp["nbest"], resp["device"], elapsed)
            self._send(200, json.dumps(resp, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")
        except Exception as e:
            import traceback
            self._send(500, json.dumps({"error": f"{e}\n{traceback.format_exc()}"}).encode(),
                       "application/json")


def main():
    # Sanity checks before serving.
    for p, name in [(CHECKPOINT, "checkpoint"), (DATA_BIN, "corpus-bin"),
                   (LANG_LIST, "lang_list.txt")]:
        if not os.path.exists(p):
            print(f"ERROR: {name} not found: {p}", file=sys.stderr)
            sys.exit(1)
    # Pre-load the default beam so the first request is instant.
    get_engine(DEFAULT_BEAM)
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"\n  Marathi -> Roman transliteration page ready.")
    print(f"  Open  http://127.0.0.1:{PORT}  in your browser.\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n(shutting down)")


if __name__ == "__main__":
    main()
