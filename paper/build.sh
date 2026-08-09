#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export SOURCE_DATE_EPOCH=1786233600
if command -v xelatex >/dev/null 2>&1; then
  xelatex -interaction=nonstopmode -halt-on-error main.tex
  xelatex -interaction=nonstopmode -halt-on-error main.tex
elif command -v tectonic >/dev/null 2>&1; then
  tectonic --keep-logs main.tex
else
  echo "XeLaTeX or Tectonic is required." >&2
  exit 1
fi
cp main.pdf wage_stagnation_decomposition_tw.pdf
python fixtounicode.py wage_stagnation_decomposition_tw.pdf
