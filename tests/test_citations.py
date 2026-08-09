from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_every_reference_is_cited_and_every_citation_exists() -> None:
    text = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    body, bibliography = text.split(r"\begin{thebibliography}", maxsplit=1)
    cited: set[str] = set()
    pattern = re.compile(r"\\cite(?:p|t)?(?:\[[^\]]*\])?(?:\[[^\]]*\])?\{([^}]+)\}")
    for match in pattern.finditer(body):
        cited.update(key.strip() for key in match.group(1).split(","))
    listed = set(re.findall(r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}", bibliography))
    assert cited == listed, {
        "uncited_references": sorted(listed - cited),
        "missing_references": sorted(cited - listed),
    }
