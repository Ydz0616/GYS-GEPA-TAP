# GYS-GEPA-TAP

GEPA-optimized attacker system prompts for a lightweight TAP jailbreak harness.

This project runs [TAP (Tree of Attacks with Pruning)](https://arxiv.org/abs/2312.02119)
as a callable function and uses [GEPA](https://arxiv.org/abs/2507.19457) to optimize
**one thing only: TAP's attacker system prompt**. The TAP loop itself — branching,
on-topic pruning, target querying, judge pruning, early stopping — is never modified.

The TAP implementation is a lightweight reimplementation of
[RICommunity/TAP](https://github.com/RICommunity/TAP): all prompt text and parsing
rules are copied verbatim from the reference repository, while the heavy transport
layer (fastchat conversation templates, HuggingFace/torch backends, wandb) is replaced
by a single OpenAI-compatible chat-completions client. GEPA is used through the
[`gepa`](https://gepa-ai.github.io/gepa/) package with a custom adapter.

> Intended for authorized red-teaming research on models you are permitted to test.
> Run artifacts under `runs/` contain real target-model responses and are gitignored.

## How the pieces fit

```
          +-----------------------------------------------+
          |  GEPA                                          |
          |    candidate = {"attacker_system_prompt": ...}  |
          +----------------------+------------------------+
                                 | system prompt
                                 v
   run_tap(system_prompt, goal, target_str, config)  ->  TAPResult
     depth x branching_factor tree, pruned by on-topic and judge scores
                                 | best payload, score, stats, trace
                                 v
          judge (LLM)  ->  5 rubric scores + actionable side_info
                                 | score + feedback
                                 v
          reflector (LLM)  ->  a new attacker system prompt
```

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Configuration

Every model call goes through one OpenAI-compatible chat-completions endpoint.

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `GEMINI_API_KEY` | — | API key (required for real runs) |
| `GYS_BASE_URL` | `https://generativelanguage.googleapis.com/v1beta/openai/` | chat-completions base URL |
| `GYS_MODEL` | `gemini-3.5-flash` | default model for every role |

Five roles — `attacker`, `target`, `evaluator` (TAP's in-loop judge and on-topic
scorer), `judge` (the GEPA-side judge), `reflector` — each take their own model,
temperature, token budget, base URL, and key environment variable. Defaults follow
the reference implementation (`ATTACK_TEMP=1`, `TARGET_TEMP=0`, evaluator at 0).

### CLI flags

| Flag | Default | Applies to |
| --- | --- | --- |
| `--model` | `gemini-3.5-flash` | sets every role at once |
| `--attacker-model`, `--target-model`, `--evaluator-model`, `--judge-model`, `--reflector-model` | inherit `--model` | per role |
| `--base-url` | `GYS_BASE_URL` | all roles |
| `--depth` | 3 | TAP tree depth |
| `--branching-factor` | 2 | children generated per surviving node |
| `--width` | 2 | nodes kept after each pruning phase |
| `--n-streams` | 1 | independent root conversations |
| `--keep-last-n` | 3 | attacker conversation turns retained |
| `--early-stop-score` | 10 | judge score that ends the run |
| `--max-metric-calls` | 12 | GEPA budget (one call = one full TAP run) |
| `--reflection-minibatch-size` | 1 | tasks per reflection step |
| `--seed` | 0 | RNG seed for pruning and splits |
| `--task-ids`, `--categories`, `--limit` | index 1 | AdvBench task selection |
| `--split` | `train` | `train`, `test`, or `heldout` |
| `--runs-dir` | `runs` | where artifacts land |

## Quickstart

One TAP run with the original (v000) attacker prompt:

```bash
.venv/bin/gys tap-run --task-ids 1 --depth 3 --branching-factor 2 --width 2
```

One GEPA optimization run (each metric call is a full TAP run, so keep the budget small):

```bash
.venv/bin/gys gepa-run --split train --task-ids 1 --max-metric-calls 12
```

Score a specific prompt version on a split, and render a run:

```bash
.venv/bin/gys eval --prompt-version 3 --split heldout
.venv/bin/gys report --run-id <run_id>
```

Add `--mock` to any command to run the whole pipeline against scripted responses with
no network calls.

## Datasets

`data/advbench_subset.csv` is the 50-goal AdvBench subset from the TAP repository.
It is split deterministically (seeded) into `train` / `test` / `heldout`; `heldout` is
never shown to the reflector and answers whether an optimized prompt generalizes.
v1 optimizes a single task (index 1) and evaluates that task plus a small heldout set.

## Outputs

Prompt versions are content-addressed and committed:

```
prompts/attacker/v000-<sha8>.md     # verbatim TAP attacker prompt
prompts/judge/v000-<sha8>.md
prompts/reflector/v000-<sha8>.md
prompts/manifest.json               # version, sha256, parent, scores, run_id
```

Each run writes `runs/<run_id>/`:

| File | Contents |
| --- | --- |
| `config.json` | resolved config, models, splits, seeds, prompt versions |
| `tap.jsonl` | one record per tree node: prompts, responses, scores, prune reason |
| `judge.jsonl` | judge calls: raw output, parsed rubric scores, `side_info` |
| `reflector.jsonl` | proposals: parent version, rendered prompt, new prompt sha |
| `gepa.jsonl` | engine events: candidates accepted/rejected, budget spent |
| `summary.json` | best candidate, per-split scores, token usage |

## Metric

v1 optimizes a single metric: `best_judge_score / 10` from the TAP run — the
jailbreak strength of the best payload the attacker prompt produced. The judge's other
dimensions (`goal_fidelity`, `stealth`, `strategy_quality`, `format_compliance`) and
efficiency statistics are recorded for every run but are not optimized until the
single-metric loop is validated.

## Tests

```bash
.venv/bin/pytest
.venv/bin/ruff check .
```

All tests use the mock backend and make no network calls.

## Plan

`docs/plan.md` is the build plan: scope, model roles, TAP fidelity notes, dataset
splits, GEPA wiring, judge rubric, versioning, tracing, and staging.
