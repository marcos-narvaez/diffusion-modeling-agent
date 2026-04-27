"""
Validators.

A generic structural-validation harness for the agent's report output.
The agent cannot finalize via task_complete until the report meets these
checks. The checks are deliberately structural, not content-specific —
they enforce that the agent has produced a complete artifact (figures,
references, sections, holdout numbers, residual discussion) without
hard-coding any particular dataset's facts.

The validators are coupled by *contract*, not by *content*:
  - the report file exists and is non-trivial
  - it embeds at least N figures
  - every figure file in figures/ is referenced in the report
  - the required section names are present
  - the agent has stated a holdout MAE/MAPE somewhere in the text
  - the residual diagnostics are discussed

Edit MIN_FIGURES, REQUIRED_SECTIONS, and HOLDOUT_PATTERN to retarget the
gate for a different report contract.
"""
import re
from pathlib import Path
from typing import List


MIN_FIGURES = 3
MIN_REPORT_LINES = 150
REQUIRED_SECTIONS = (
    "Executive Summary",
    "Data Overview",
    "Modeling",
    "Validation",
    "Forecast",
)
# A "holdout" disclosure: the agent must state a holdout error metric in
# the text — MAE, MAPE, or RMSE accompanied by a number.
HOLDOUT_PATTERN = re.compile(
    r"\b(MAE|MAPE|RMSE)\b[^\n]{0,80}?(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
RESIDUAL_PATTERN = re.compile(r"\bresidual", re.IGNORECASE)
NUMERIC_FORECAST_PATTERN = re.compile(r"\b\d{2,}\b")


def validate_report(workspace: Path,
                    report_name: str = "report.tex",
                    figures_dir: str = "figures") -> List[str]:
    """
    Run the full structural validation on the report and return a list of
    failure strings. An empty list means the gate passed.
    """
    failures: List[str] = []
    report_path = workspace / report_name
    figures_path = workspace / figures_dir

    # ── 1. Report exists and is non-trivial ──
    if not report_path.exists():
        failures.append(f"{report_name} does not exist in workspace/")
        return failures
    text = report_path.read_text()
    line_count = len(text.splitlines())
    if line_count < MIN_REPORT_LINES:
        failures.append(
            f"{report_name} has only {line_count} lines; expected >= {MIN_REPORT_LINES}"
        )

    # ── 2. Embedded figures ──
    graphics_refs = re.findall(r"\\includegraphics", text)
    if len(graphics_refs) < MIN_FIGURES:
        failures.append(
            f"{report_name} embeds only {len(graphics_refs)} figure(s); "
            f"expected >= {MIN_FIGURES}"
        )

    # ── 3. Required section headers ──
    lower = text.lower()
    for header in REQUIRED_SECTIONS:
        if header.lower() not in lower:
            failures.append(f"{report_name} missing section containing '{header}'")

    # ── 4. Holdout error disclosure ──
    if not HOLDOUT_PATTERN.search(text):
        failures.append(
            f"{report_name} does not state a holdout error metric (MAE, MAPE, or RMSE) with a value"
        )

    # ── 5. Residual diagnostics ──
    if not RESIDUAL_PATTERN.search(text):
        failures.append(
            f"{report_name} does not discuss residual diagnostics"
        )

    # ── 6. Figures directory and cross-reference ──
    if not figures_path.exists():
        failures.append(f"{figures_dir}/ directory does not exist")
    else:
        pngs = sorted(figures_path.glob("*.png"))
        if len(pngs) < MIN_FIGURES:
            failures.append(
                f"{figures_dir}/ contains only {len(pngs)} PNG(s); "
                f"expected >= {MIN_FIGURES}"
            )
        for png in pngs:
            if png.name not in text:
                failures.append(
                    f"{figures_dir}/{png.name} exists but is not referenced in {report_name}"
                )

    # ── 7. Non-standard LaTeX commands ──
    if r"\LL" in text:
        failures.append(
            r"\LL is not a standard LaTeX command; use $\mathrm{LL}$ or plain LL"
        )

    # ── 8. At least some numeric content (sanity) ──
    if len(NUMERIC_FORECAST_PATTERN.findall(text)) < 5:
        failures.append(
            f"{report_name} contains very few numeric values; results should be quantitative"
        )

    return failures
