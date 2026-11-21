#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
venv_path="$HOME/.venvs/indicxlict"
uv_bin="$(command -v uv || true)"
if [[ -z "$uv_bin" && -x "$HOME/.local/bin/uv" ]]; then
  uv_bin="$HOME/.local/bin/uv"
fi

mkdir -p "$(dirname "$venv_path")"
if [[ -n "$uv_bin" ]]; then
  "$uv_bin" python install 3.10
  if [[ ! -x "$venv_path/bin/python" ]]; then
    "$uv_bin" venv --python 3.10 "$venv_path"
  fi
  "$uv_bin" pip install --python "$venv_path/bin/python" \
    pip==24.0 setuptools==59.5.0 wheel
  "$uv_bin" pip install --python "$venv_path/bin/python" \
    torch==2.0.1 torchaudio==2.0.2 numpy==1.26.4 \
    --index-url https://download.pytorch.org/whl/cpu
  "$uv_bin" pip install --python "$venv_path/bin/python" \
    fairseq==0.12.2 indic-nlp-library==0.92
else
  if ! command -v python3.10 >/dev/null 2>&1; then
    echo "Python 3.10 or uv is required in WSL. Install uv from https://docs.astral.sh/uv/getting-started/installation/, then rerun ./setup_wsl.sh." >&2
    exit 1
  fi
  if [[ ! -x "$venv_path/bin/python" ]]; then
    python3.10 -m venv "$venv_path"
  fi
  "$venv_path/bin/python" -m pip install --upgrade pip==24.0 setuptools==59.5.0 wheel
  "$venv_path/bin/python" -m pip install \
    torch==2.0.1 torchaudio==2.0.2 numpy==1.26.4 \
    --index-url https://download.pytorch.org/whl/cpu
  "$venv_path/bin/python" -m pip install fairseq==0.12.2 indic-nlp-library==0.92
fi

echo
echo "WSL environment is ready. Activate it with:"
echo "source '$venv_path/bin/activate'"
echo "Then run from the project root:"
echo "python '$project_root/marathi_beam_search.py' 'namaskar maharashtra' --beam 10 --nbest 5"
