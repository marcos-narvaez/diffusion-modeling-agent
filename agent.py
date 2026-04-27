#!/usr/bin/env python3
"""
diffusion-modeling-agent
========================

A hand-rolled ReAct orchestrator that drives a Claude model through the
full pipeline of fitting an aggregate adoption-timing model: explore the
data, fit a sequence of nested and non-nested specifications, validate on
holdout, write a report, and survive a critic + validator gate before
declaring done.

Architecture
------------

  +------------------+        system prompt + task brief
  |   Orchestrator   |---------------------------------+
  |   (this file)    |                                 |
  +--------+---------+                                 |
           |                                           v
           |   tool calls          +---------------------------+
           +---------------------> |        Claude             |
                                   |  (Anthropic SDK direct)   |
                                   +-------------+-------------+
                                                 |
                                                 v
                                   +---------------------------+
                                   |        Tool Router        |
                                   |  run_python   read_file   |
                                   |  write_file   think       |
                                   |  task_complete            |
                                   +-------------+-------------+
                                                 |
                                                 v
                                   +---------------------------+
                                   |   Critic + Validator gate |
                                   |   (peer review then       |
                                   |    structural validation) |
                                   +---------------------------+

The critic is a separate Claude call that reads the report draft and
returns a JSON list of issues. Two critic rounds are allowed before the
gate falls through to the structural validator. Only when both clear may
task_complete actually terminate the loop.

Usage
-----

  export ANTHROPIC_API_KEY="sk-ant-..."
  python agent.py
"""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import anthropic

from system_prompt import SYSTEM_PROMPT
from validators import validate_report


# ── Configuration ──────────────────────────────────────────────────────────

MODEL = os.environ.get("AGENT_MODEL", "claude-sonnet-4-6")
MAX_TOKENS = 8000
MAX_TURNS = 150
WORK_DIR = Path("./workspace")
LOG_FILE = Path("./agent_log.jsonl")

# Source data lives outside the workspace; the orchestrator copies it in.
PROJECT_DIR = Path(__file__).resolve().parent
DATA_SRC = PROJECT_DIR / "data" / "adoption_synthetic.csv"
META_SRC = PROJECT_DIR / "data" / "adoption_synthetic.meta.json"


# ── Tool Definitions ───────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "run_python",
        "description": (
            "Execute a Python script in an isolated subprocess. The script has "
            "access to: pandas, numpy, scipy, matplotlib, and the dataset at "
            "./data/adoption_synthetic.csv. Returns stdout and stderr. Use this "
            "for ALL computation: data loading, model fitting, plotting, LaTeX "
            "generation. Each call runs in a fresh process — variables do NOT "
            "persist between calls. Re-import and re-load at the top of every "
            "script."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Complete Python script to execute.",
                },
                "purpose": {
                    "type": "string",
                    "description": "Brief description of what this script does (for logging).",
                },
            },
            "required": ["code", "purpose"],
        },
    },
    {
        "name": "read_file",
        "description": (
            "Read the contents of a file in the workspace directory. Useful for "
            "inspecting outputs, intermediate CSV results, and the report draft."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path within ./workspace/",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Write content to a file in the workspace directory. Use for saving "
            "notes, intermediate results, the report.tex draft, and other "
            "non-Python outputs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path within ./workspace/",
                },
                "content": {
                    "type": "string",
                    "description": "File content to write.",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "think",
        "description": (
            "Write down your reasoning before taking an action. Use this when "
            "you are about to make an important decision (model selection, "
            "parameter interpretation, report structure) or when you need to "
            "work through a subtle calculation. The contents are recorded in "
            "the conversation but no code runs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "thought": {"type": "string", "description": "Your reasoning."}
            },
            "required": ["thought"],
        },
    },
    {
        "name": "task_complete",
        "description": (
            "Signal that the modeling task is complete. The orchestrator will "
            "first run a critic review of the report and then a structural "
            "validation gate. Only when both pass will the loop actually "
            "terminate."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Final summary of models built, selected model, and key findings.",
                }
            },
            "required": ["summary"],
        },
    },
]


# ── Tool Implementations ──────────────────────────────────────────────────

def execute_python(code: str) -> str:
    """Run a script in a subprocess inside the workspace dir. Return stdout+stderr."""
    script_path = (WORK_DIR / "_temp_script.py").resolve()
    preamble = f"""
import warnings
warnings.filterwarnings('ignore')
import os
os.chdir('{WORK_DIR.resolve()}')
os.makedirs('figures', exist_ok=True)
import matplotlib
matplotlib.use('Agg')
import numpy as np
np.set_printoptions(suppress=True, precision=4)
"""
    script_path.write_text(preamble + code)
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=600,
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += f"\n[STDERR]: {result.stderr}"
        if result.returncode != 0:
            output += f"\n[EXIT CODE]: {result.returncode}"
        return output[:15000]
    except subprocess.TimeoutExpired:
        return "[ERROR]: Script timed out after 600 seconds."
    except Exception as e:
        return f"[ERROR]: {e}"


def read_file(path: str) -> str:
    full_path = WORK_DIR / path
    if not full_path.exists():
        return f"[ERROR]: File not found: {path}"
    try:
        return full_path.read_text()[:20000]
    except Exception as e:
        return f"[ERROR]: {e}"


def write_file(path: str, content: str) -> str:
    full_path = WORK_DIR / path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        full_path.write_text(content)
        return f"[OK]: Written {len(content)} bytes to {path}"
    except Exception as e:
        return f"[ERROR]: {e}"


def handle_tool_call(tool_name: str, tool_input: dict) -> str:
    if tool_name == "run_python":
        return execute_python(tool_input["code"])
    if tool_name == "read_file":
        return read_file(tool_input["path"])
    if tool_name == "write_file":
        return write_file(tool_input["path"], tool_input["content"])
    if tool_name == "think":
        return f"[Thought recorded]: {tool_input.get('thought', '')}"
    if tool_name == "task_complete":
        return "[TASK COMPLETE]"
    return f"[ERROR]: Unknown tool: {tool_name}"


# ── Logging ────────────────────────────────────────────────────────────────

def log_turn(turn_num: int, role: str, content: str) -> None:
    entry = {
        "timestamp": datetime.now().isoformat(),
        "turn": turn_num,
        "role": role,
        "content": content[:500],
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def banner(turn: int, msg: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  Turn {turn}: {msg}")
    print(f"{'=' * 60}")


# ── Rate-limit-aware API call ─────────────────────────────────────────────

def call_claude_with_retry(client: anthropic.Anthropic, **kwargs):
    max_retries = 5
    for attempt in range(max_retries):
        try:
            return client.messages.create(**kwargs)
        except anthropic.RateLimitError:
            if attempt == max_retries - 1:
                raise
            wait = 60 * (2 ** attempt)
            print(f"[Rate limit hit, waiting {wait}s before retry {attempt + 2}]")
            time.sleep(wait)
        except anthropic.APIError as e:
            if attempt == max_retries - 1:
                raise
            wait = 10 * (2 ** attempt)
            print(f"[API error, waiting {wait}s before retry: {e}]")
            time.sleep(wait)


# ── Critic ────────────────────────────────────────────────────────────────

CRITIC_SYSTEM_PROMPT = """You are a rigorous peer reviewer examining a draft
report on aggregate adoption-timing models. Your job is to identify
specific, actionable errors and weaknesses. Be precise and demanding.

## KNOWN FAILURE MODES TO CHECK

1. Inverted heterogeneity interpretation: in a Gamma(r, alpha) mixture, r < 1
   is HIGH heterogeneity (most agents have low hazards, with a thin tail of
   fast adopters). r > 1 is LOW heterogeneity. If the report writes r < 1 as
   "homogeneous" or "low heterogeneity," that is a critical error.

2. Marginal-effect arithmetic with scaled covariates: if a covariate was
   scaled (e.g., divided by its mean), the fitted beta applies to the SCALED
   variable. The percent change in hazard for a 1-unit change in the original
   covariate is exp(beta / scale_factor) - 1, NOT exp(beta) - 1. Flag any
   sloppy arithmetic.

3. Implausible seasonal coefficients: if a buildup/peak/recovery decomposition
   yields recovery > peak, or buildup strongly negative, the seasonal
   specification is mis-shaped.

4. Missing residual discussion: if the report claims "exceptional fit" without
   inspecting residuals for drift, heteroskedasticity, or autocorrelation,
   that is a significant omission.

5. Missing figure references: every PNG in figures/ must have an
   \\includegraphics in the report.

6. Non-standard LaTeX (\\LL, undefined commands, missing packages).

7. Forecast plausibility: forecasts that diverge sharply from the most recent
   observed weeks without a justified covariate change should be challenged.

## GENERAL CHECKLIST

- Each section follows logically from the previous.
- Parameter interpretations are internally consistent.
- Model selection is justified on story plus BIC, not BIC alone.
- All claims are quantitatively backed up.
- Executive summary matches the results section.

## OUTPUT FORMAT

Return a JSON object with this structure exactly:
{
  "issues": [
    {
      "severity": "critical" | "important" | "minor",
      "location": "section name or paragraph description",
      "issue": "precise description of the problem",
      "suggestion": "specific revision instruction"
    }
  ],
  "overall_assessment": "brief paragraph on overall quality and what must be done"
}

Be specific. "The covariate interpretation is wrong" is not enough — explain
exactly what the report said, what it should have said, and what the author
needs to redo. If there are no issues in a category, omit those entries.
Do not invent problems that don't exist.
"""


def run_critic(client: anthropic.Anthropic, report_text: str) -> dict:
    print("\n  [Critic]: Sending draft to critic agent...")
    prompt = f"""Please review this draft report and identify all issues
according to your instructions.

<report>
{report_text}
</report>

Return your review as JSON."""
    try:
        response = call_claude_with_retry(
            client,
            model=MODEL,
            max_tokens=8000,
            system=CRITIC_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text if response.content else "{}"
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {
            "issues": [{
                "severity": "important",
                "location": "full report",
                "issue": "Critic returned non-JSON response",
                "suggestion": raw[:2000],
            }],
            "overall_assessment": raw[:500],
        }
    except Exception as e:
        print(f"  [Critic ERROR]: {e}")
        return {"issues": [], "overall_assessment": f"Critic failed: {e}"}


def format_critic_feedback(review: dict) -> str:
    issues = review.get("issues", [])
    assessment = review.get("overall_assessment", "")
    if not issues:
        return (
            "A reviewer examined your draft and found no significant issues. "
            "You may proceed to finalize and call task_complete."
        )
    lines = [
        "A reviewer has examined your draft and identified the following "
        "issues that MUST be addressed before the report can be finalized:\n"
    ]
    for i, issue in enumerate(issues, 1):
        severity = issue.get("severity", "unknown").upper()
        location = issue.get("location", "unknown")
        problem = issue.get("issue", "")
        suggestion = issue.get("suggestion", "")
        lines.append(
            f"{i}. [{severity}] In {location}:\n"
            f"   Problem: {problem}\n"
            f"   Fix: {suggestion}\n"
        )
    if assessment:
        lines.append(f"\nOverall assessment: {assessment}")
    lines.append(
        "\nRevise the report (and any model code if parameters need recomputing) "
        "to address each issue above, then call task_complete."
    )
    return "\n".join(lines)


# ── Main Loop ──────────────────────────────────────────────────────────────

def setup_workspace() -> None:
    WORK_DIR.mkdir(exist_ok=True)
    (WORK_DIR / "figures").mkdir(exist_ok=True)
    (WORK_DIR / "data").mkdir(exist_ok=True)
    import shutil
    if not (WORK_DIR / "data" / "adoption_synthetic.csv").exists():
        if not DATA_SRC.exists():
            print(f"[ERROR] Source data not found at {DATA_SRC}.")
            print("        Run `python data/generate_synthetic.py` first.")
            sys.exit(1)
        shutil.copy(DATA_SRC, WORK_DIR / "data" / "adoption_synthetic.csv")
    if not (WORK_DIR / "data" / "adoption_synthetic.meta.json").exists() and META_SRC.exists():
        shutil.copy(META_SRC, WORK_DIR / "data" / "adoption_synthetic.meta.json")


def run_agent() -> None:
    setup_workspace()
    client = anthropic.Anthropic()

    messages = [{
        "role": "user",
        "content": (
            "Begin the autonomous modeling workflow. Start by loading and "
            "exploring the dataset at data/adoption_synthetic.csv (and the "
            "metadata file alongside it), then systematically build models as "
            "described in your instructions. Call task_complete only after the "
            "report draft and figures are written."
        ),
    }]

    task_done = False
    turn = 0
    total_input_tokens = 0
    total_output_tokens = 0
    critic_rounds_done = 0
    MAX_CRITIC_ROUNDS = 2

    print("\n" + "=" * 60)
    print(" diffusion-modeling-agent — autonomous modeling run".center(60))
    print("=" * 60)

    while not task_done and turn < MAX_TURNS:
        turn += 1
        banner(turn, "Calling Claude...")
        time.sleep(5)  # proactive throttle
        try:
            response = call_claude_with_retry(
                client,
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )
        except Exception as e:
            print(f"[API ERROR]: {e}")
            break

        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens

        assistant_content = response.content
        messages.append({"role": "assistant", "content": assistant_content})

        for block in assistant_content:
            if block.type == "text":
                print(f"\n[Claude]: {block.text[:300]}...")
                log_turn(turn, "assistant", block.text)

        has_tool_use = any(b.type == "tool_use" for b in assistant_content)
        if has_tool_use:
            tool_results = []
            for block in assistant_content:
                if block.type != "tool_use":
                    continue
                tool_name = block.name
                tool_input = block.input
                tool_id = block.id

                purpose = tool_input.get("purpose", tool_name)
                print(f"\n  Tool: {tool_name}")
                print(f"  Purpose: {purpose}")

                if tool_name == "task_complete":
                    tool_result = None

                    # Round 1+: critic review of the draft
                    if critic_rounds_done < MAX_CRITIC_ROUNDS:
                        critic_rounds_done += 1
                        print(f"\n  [Critic round {critic_rounds_done}/{MAX_CRITIC_ROUNDS}]")
                        report_text = read_file("report.tex")
                        time.sleep(5)
                        review = run_critic(client, report_text)
                        feedback = format_critic_feedback(review)
                        num_issues = len(review.get("issues", []))
                        print(f"  [Critic]: Found {num_issues} issue(s)")
                        if num_issues > 0:
                            tool_result = feedback
                        else:
                            critic_rounds_done = MAX_CRITIC_ROUNDS

                    if tool_result is None:
                        # Validator gate
                        failures = validate_report(WORK_DIR)
                        if failures:
                            print(f"\n  VALIDATION FAILED — {len(failures)} issue(s):")
                            for vf in failures:
                                print(f"    - {vf}")
                            tool_result = (
                                "Validation failed. The following issues must "
                                "be fixed before task_complete is allowed:\n"
                                + "\n".join(f"- {vf}" for vf in failures)
                                + "\n\nAddress each issue, then call task_complete again."
                            )
                        else:
                            print(f"\n  TASK COMPLETE (critic + validation passed)")
                            print(f"  {tool_input.get('summary', '')[:200]}")
                            task_done = True
                            tool_result = "[TASK COMPLETE]"
                else:
                    tool_result = handle_tool_call(tool_name, tool_input)
                    preview = tool_result[:200].replace("\n", " ")
                    print(f"  Result: {preview}...")

                log_turn(turn, f"tool:{tool_name}", tool_result[:500])
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": tool_result,
                })

            if not task_done:
                messages.append({"role": "user", "content": tool_results})

        elif response.stop_reason == "end_turn" and not has_tool_use:
            print("\n[Agent]: Claude stopped without tool use. Nudging...")
            messages.append({
                "role": "user",
                "content": (
                    "Continue with the next step in the modeling workflow. "
                    "If you're done, call task_complete."
                ),
            })

    print("\n" + "=" * 60)
    print(" Agent run complete".center(60))
    print("=" * 60)
    print(f"  Turns: {turn}")
    print(f"  Input tokens:  {total_input_tokens:,}")
    print(f"  Output tokens: {total_output_tokens:,}")
    print(f"  Workspace: {WORK_DIR.resolve()}")
    print(f"  Log: {LOG_FILE.resolve()}")

    outputs = list(WORK_DIR.glob("**/*"))
    print(f"\n  Files generated ({len(outputs)}):")
    for f in sorted(outputs):
        if f.is_file() and f.name != "_temp_script.py":
            print(f"    {f.relative_to(WORK_DIR)}")


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: set ANTHROPIC_API_KEY environment variable.")
        print("  export ANTHROPIC_API_KEY='sk-ant-...'")
        sys.exit(1)
    run_agent()
