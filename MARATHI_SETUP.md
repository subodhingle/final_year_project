# IndicXlit Marathi setup

This workspace contains the official AI4Bharat IndicXlit source and both v1.0 multilingual checkpoints. Marathi uses language code `mr` in the inference interface and is included in both checkpoints.

## Files

- `models/en-indic/transformer/indicxlit.pt` — Roman to Indic checkpoint
- `models/indic-en/transformer/indicxlit.pt` — Indic to Roman checkpoint
- `models/*/corpus-bin/` — model dictionaries needed by Fairseq
- `inference/` — upstream CLI and Python inference code

The downloaded archives and checkpoints are ignored by Git (`*.zip`, `*.pt`), so they stay local and do not inflate source commits.

## Windows PowerShell

From the project folder, set up the local environment with:

```powershell
cd E:\indicxlict
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
```

The setup script creates `.venv` with Python 3.10, CPU PyTorch, Fairseq 0.12.2, and Indic NLP. In VS Code, choose `.venv\Scripts\python.exe` if it asks you to select an interpreter. Press **F5**, select **IndicXlit Marathi beam search**, and enter text, direction, beam width, and number of candidates when prompted.

You can also run the launcher in the VS Code PowerShell terminal without Bash line continuations:

```powershell
.\.venv\Scripts\python.exe .\marathi_beam_search.py "namaskar maharashtra" --direction roman-to-marathi --beam 10 --nbest 5
.\.venv\Scripts\python.exe .\marathi_beam_search.py "नमस्कार महाराष्ट्र" --direction marathi-to-roman --beam 10 --nbest 5
```

The launcher uses the project `.venv`; don’t select the system Python 3.13 interpreter. This Windows setup runs with Python 3.10, CPU PyTorch 2.0.1, and Fairseq 0.12.2. Fairseq can be difficult to build on some Windows machines, so the setup script pins this compatible runtime.

## Linux or WSL

The Windows `.venv` is separate from WSL. In your WSL shell, go to the project root and set up WSL's own Python environment:

```bash
cd /mnt/e/indicxlict
bash ./setup_wsl.sh
source ~/.venvs/indicxlict/bin/activate
python marathi_beam_search.py "namaskar maharashtra" --direction roman-to-marathi --beam 10 --nbest 5
```

The WSL setup uses Python 3.10, CPU PyTorch, Fairseq 0.12.2, and Indic NLP. The script uses `uv` to install Python 3.10 when the distro only has Python 3.12. If VS Code is connected to WSL, select `~/.venvs/indicxlict/bin/python` as its interpreter before pressing **F5**. The commands below are Bash commands and need to run in a WSL/Linux terminal.

### Manual WSL setup

If you prefer installing dependencies manually, create the Python 3.10 venv and install the same pinned packages:

```bash
uv python install 3.10
uv venv --python 3.10 ~/.venvs/indicxlict
uv pip install --python ~/.venvs/indicxlict/bin/python pip==24.0 setuptools==59.5.0 wheel
uv pip install --python ~/.venvs/indicxlict/bin/python torch==2.0.1 torchaudio==2.0.2 numpy==1.26.4 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python ~/.venvs/indicxlict/bin/python fairseq==0.12.2 indic-nlp-library==0.92
```

## Roman to Marathi

From the repository root, prepare one word per line in a UTF-8 file, for example `input.txt`:

```text
namaskar
maharashtra
```

Then run:

```bash
cd inference/cli/en-indic
mkdir -p output
fairseq-interactive "../../../models/en-indic/corpus-bin" \
  --path "../../../models/en-indic/transformer/indicxlit.pt" \
  --task translation_multi_simple_epoch \
  --beam 10 --nbest 1 \
  --source-lang en --target-lang mr \
  --encoder-langtok tgt --lang-dict lang_list.txt \
  --lang-pairs en-as,en-bn,en-brx,en-gom,en-gu,en-hi,en-kn,en-ks,en-mai,en-ml,en-mni,en-mr,en-ne,en-or,en-pa,en-sa,en-sd,en-si,en-ta,en-te,en-ur \
  --input ../../../input.txt > output/en_mr.txt
```

## Marathi to Roman

Prepare Marathi words, one per line, in a UTF-8 file such as `input_mr.txt`, then run:

```bash
cd inference/cli/indic-en
mkdir -p output
fairseq-interactive "../../../models/indic-en/corpus-bin" \
  --path "../../../models/indic-en/transformer/indicxlit.pt" \
  --task translation_multi_simple_epoch \
  --beam 10 --nbest 1 \
  --source-lang mr --target-lang en \
  --encoder-langtok tgt --lang-dict lang_list.txt \
  --lang-pairs as-en,bn-en,brx-en,gom-en,gu-en,hi-en,kn-en,ks-en,mai-en,ml-en,mni-en,mr-en,ne-en,or-en,pa-en,sa-en,sd-en,si-en,ta-en,te-en,ur-en \
  --input ../../../input_mr.txt > output/mr_en.txt
```

IndicXlit was trained with words as inputs. For sentence text, split into whitespace-separated words before inference and join the top predictions afterward. Fairseq's `D-` lines in the output contain decoded predictions.

## Beam search from Python

`marathi_beam_search.py` connects the downloaded checkpoints to Fairseq's beam decoder and accepts text directly. It splits whitespace-separated words, formats each word as space-separated characters, requests the top hypotheses, and prints the candidates with their model scores.

```bash
python marathi_beam_search.py "namaskar maharashtra" --direction roman-to-marathi --beam 10 --nbest 5
python marathi_beam_search.py "नमस्कार महाराष्ट्र" --direction marathi-to-roman --beam 10 --nbest 5
```

`--beam` controls how many partial sequences Fairseq explores while decoding. `--nbest` controls how many completed candidates are returned, and must be no greater than the beam width. For example, `--beam 10 --nbest 5` explores with a beam of ten and displays five ranked options for each word. The exact candidates depend on the checkpoint and runtime; scores are Fairseq log scores, where a higher (less negative) score ranks ahead.

## Optional rescoring

The upstream rescoring dictionaries are a separate ~850 MB download and are not included. The checkpoints run without them; rescoring can improve candidate ranking for supported languages. The project release links are in the main README.
