# diffusion-modeling-agent

A hand-rolled ReAct agent that autonomously builds, fits, validates, and reports statistical models for aggregate adoption-timing data. Direct Anthropic SDK, no LangChain or LlamaIndex, five tools, a 150-turn cap, and a critic-then-validator gate that has to clear before the agent can declare done. End-to-end runs converge in roughly 90-100 turns on the included synthetic dataset.

## Architecture

Five tools, one orchestrator, one critic, one structural validator. The orchestrator owns the message log, the tool router, the rate-limit retry, and the gate state. The critic is a separate Claude call that reads the draft report and returns structured JSON issues. The validator is local Python — no LLM, just file-system checks and regex.

```
+------------------+       system prompt + task brief
|   Orchestrator   |--------------------------------+
|   (agent.py)     |                                |
+--------+---------+                                |
         |   tool calls                             v
         |                            +----------------------------+
         +--------------------------> |          Claude            |
                                      |   (Anthropic SDK direct)   |
                                      +-------------+--------------+
                                                    |
                                                    v
                                      +----------------------------+
                                      |        Tool Router         |
                                      |  run_python   read_file    |
                                      |  write_file   think        |
                                      |  task_complete             |
                                      +-------------+--------------+
                                                    |
                            (only when task_complete is invoked)
                                                    v
                                      +----------------------------+
                                      |  Critic + Validator gate   |
                                      |  - peer-review JSON pass   |
                                      |  - structural validators   |
                                      +----------------------------+
```

The five tools:

- `run_python` — executes a complete script in an isolated subprocess inside `./workspace/`. Each call is a fresh process; variables do not persist. Returns combined stdout and stderr, capped at 15 KB. This is the agent's only computation channel.
- `read_file` — reads a workspace-relative file, capped at 20 KB.
- `write_file` — writes a workspace-relative file (creates parent dirs).
- `think` — records reasoning into the conversation without running code. Used as a scratchpad before parameter interpretation, model selection, and prose drafting.
- `task_complete` — does NOT terminate the loop. It triggers the gate. The gate runs the critic first (up to two rounds, returning JSON-formatted issues that come back as the next user message), then runs the structural validator (figure references, section headers, residual mention, holdout-error disclosure). Only if both clear does the loop actually exit.

## What it does

Input: a tabular CSV of weekly adoption counts plus a time-varying covariate plus a binary seasonal indicator, with a market-size constant `N` provided alongside in a metadata file.

The agent fits a sequence of aggregate timing-to-event models — Geometric, Pareto II, Burr XII — and progressively adds heterogeneity, individual-level duration dependence, and proportional-hazards covariates. It selects on BIC subject to story constraints, runs a holdout test (refits on a truncated window and forecasts the rest), inspects residuals, and produces a LaTeX report with at least three embedded figures.

Output: `workspace/report.tex`, the figure directory, and a JSONL turn-by-turn log.

The validator enforces structure before completion: report exists, has at least the required section headers, every figure on disk is referenced in the LaTeX, residual diagnostics are present, and a holdout MAE/MAPE/RMSE value is stated in prose. The agent cannot lie its way to "done" — at least, not without lying in a structurally-detectable way.

## Why hand-rolled

LangChain, LlamaIndex, and smolagents would each have added a layer of indirection between me and the message log, the tool boundaries, and the loop control. The interesting work in this kind of agent is precisely there: how to format tool results so the model integrates them cleanly, when to interject a critic call, how to keep the validator gate from being gameable, what to log per turn for debugging. With the Anthropic SDK directly I can read every byte that crosses the wire and tune any of those decisions. The framework abstractions hide exactly the things I want to see.

The cost is reinventing rate-limit retry, message accumulation, and tool dispatch. That cost is small (the orchestrator is under 500 lines including comments) and the visibility is worth it.

## Quickstart

```bash
git clone <this repo>
cd diffusion-modeling-agent
pip install -e .              # installs anthropic, numpy, scipy, pandas, matplotlib

python data/generate_synthetic.py     # writes data/adoption_synthetic.{csv,meta.json}

export ANTHROPIC_API_KEY="sk-ant-..."
python agent.py
```

The agent writes a turn-by-turn log to `agent_log.jsonl`, copies the dataset into `workspace/data/`, and on a successful run produces `workspace/report.tex` plus figures under `workspace/figures/`.

To run the model files directly without invoking the agent:

```bash
cd models
python model_09_burr_xii_exposure_season_FINAL.py
```

To run the test suite:

```bash
pytest
```

A redacted excerpt of an actual agent run is in [`examples/sample_run.md`](examples/sample_run.md).

## Honest limitations

These are real, and the right thing to do is enumerate them rather than paper over them.

- **Single model end to end.** The orchestrator and the critic both call the same Sonnet/Opus model. There is no Haiku-as-fast-validator fallback. A cheaper critic round on a smaller model would be a sensible upgrade and is not implemented.
- **No prompt caching.** The system prompt is large (~6 KB) and is sent uncached on every turn. Wiring `cache_control` blocks would cut input cost meaningfully and is a TODO.
- **No async tool execution.** The tool router is synchronous. The agent never issues parallel tool calls; nothing in the architecture would prevent it, but I haven't built the dispatcher.
- **No token-cost tracking with prices.** The orchestrator counts input and output tokens but does not multiply by the model's price card. A run on the synthetic dataset uses roughly 1-2M input tokens depending on how chatty the loop gets — non-trivial cost, not currently surfaced.
- **Brittle string-match validators.** The structural validator uses regex on the LaTeX source. A determined model could satisfy the regex without satisfying the spirit of the check (e.g., type the word "residual" in a footnote without producing a residual plot). I have not seen this happen in practice, but it is a real gap that a more careful semantic validator would close.
- **Critic gate is bounded at two rounds.** If the critic keeps finding issues after two rounds, the orchestrator falls through to the structural validator and lets the model declare done if structure passes. This is a deliberate trade — it avoids infinite loops on irreducible disagreements — but it means the critic is advisory, not blocking.
- **Single dataset shape.** The model files assume the data has the shape produced by `data/generate_synthetic.py`: one time-varying continuous covariate plus one binary seasonal indicator, integer adoptions per week, with the last 4 weeks withheld. Adapting to a different schema requires editing the model files.
- **No retries on Python exceptions inside `run_python`.** If a script raises, the traceback comes back to the model and the model decides whether to retry. The orchestrator does not auto-correct.
- **Workspace persistence between runs.** The orchestrator does not clear `workspace/` between runs. An aborted run can leave stale files that influence the next run's validator pass. Delete `workspace/` to start clean.

## Repo layout

```
diffusion-modeling-agent/
  README.md
  LICENSE
  pyproject.toml
  .gitignore
  agent.py                    orchestrator
  system_prompt.py            embedded domain knowledge passed every turn
  validators.py               structural validator gate
  data/
    generate_synthetic.py     produces adoption_synthetic.{csv,meta.json}
    adoption_synthetic.csv    generated; in .gitignore? no — included
  models/
    _shared.py                load_data, fit_model, print_results
    model_01_geometric.py     baseline: constant hazard
    model_02_pareto_ii.py     exponential-gamma mixture
    model_03_burr_xii.py      weibull-gamma mixture (duration dependence)
    model_04_pareto_ii_exposure.py
    model_05_burr_xii_exposure.py
    model_06_burr_xii_log_exposure.py        (expected to fit poorly)
    model_07_burr_xii_lagged_exposure.py
    model_08_burr_xii_exposure_season_dummy.py
    model_09_burr_xii_exposure_season_FINAL.py    reference final spec
    model_10_burr_xii_lagged_exposure_season.py
    model_11_burr_xii_both_exposure_season.py     (multicollinearity)
    model_12_burr_xii_exposure_season_two.py
    model_13_burr_xii_exposure_buildup_peak_recovery.py
    model_14_burr_xii_lagged_exposure_season.py
    model_15_holdout_validation.py
    model_16_burr_xii_exposure_season_2segment.py
  examples/
    sample_run.md             redacted excerpt of an actual run
  tests/
    test_models.py            sanity tests; pytest
```

## References

- F. M. Bass (1969). "A new product growth model for consumer durables." *Management Science*.
- D. R. Cox (1972). "Regression models and life-tables." *JRSS B*. (Proportional hazards — the construction used here for time-varying covariates.)
- I. W. Burr (1942). "Cumulative frequency functions." *Annals of Mathematical Statistics*. (The Burr XII family used here for the survival function.)
- J. F. Lawless (2003). *Statistical Models and Methods for Lifetime Data*, 2nd ed., Wiley. (General reference for parametric survival models, MLE on the survivor-augmented likelihood, and aggregate-data estimation.)

## Author

Marcos Narváez.
