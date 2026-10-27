"""Command-line entry point: tap-run, gepa-run, eval, report.

All commands accept the same transport/model/TAP flags and a ``--mock`` switch
that runs the whole pipeline offline. Every run writes a timestamped directory
under ``runs/`` containing ``config.json``, JSONL traces, and ``summary.json``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import gepa

from gys.config import RunConfig
from gys.datasets import select_tasks
from gys.gepa_adapter import COMPONENT, TAPGEPAAdapter
from gys.judge import Judge
from gys.llm import build_client
from gys.reflector import make_reflection_lm, reflection_prompt_template
from gys.tap.loop import run_tap
from gys.tap.prompts import seed_attacker_system_prompt
from gys.tap.versioning import PromptStore
from gys.tracing import JSONLSink, new_run_id


def _add_common_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="JSON config file; flags override it.")
    parser.add_argument("--mock", action="store_true", help="Run offline with a mock LLM.")
    parser.add_argument("--model", help="Override the model for every role.")
    parser.add_argument("--base-url", help="Override the chat-completions base URL.")
    parser.add_argument("--api-key-env", help="Env var holding the API key.")
    parser.add_argument("--runs-dir", help="Directory for run artifacts (default runs/).")
    parser.add_argument("--prompts-dir", help="Directory of versioned prompts.")
    # Per-role model overrides.
    for role in ("attacker", "target", "evaluator", "judge", "reflector"):
        parser.add_argument(f"--{role}-model", help=f"Override the {role} model.")
    # TAP knobs.
    parser.add_argument("--depth", type=int)
    parser.add_argument("--branching-factor", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--n-streams", type=int)
    parser.add_argument("--keep-last-n", type=int)
    parser.add_argument("--max-attack-attempts", type=int)
    parser.add_argument("--early-stop-score", type=int)
    parser.add_argument("--seed", type=int)
    # Data selection.
    parser.add_argument("--task-ids", help="Comma-separated AdvBench row ids.")
    parser.add_argument("--split", choices=["train", "test", "heldout"])
    parser.add_argument("--categories", help="Comma-separated category filter.")
    parser.add_argument("--limit", type=int)
    # GEPA budget.
    parser.add_argument("--max-metric-calls", type=int)


def _int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(x) for x in value.split(",") if x.strip())


def _str_tuple(value: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in value.split(",") if x.strip())


def _config_from_args(args: argparse.Namespace) -> RunConfig:
    cfg = RunConfig.from_file(args.config) if args.config else RunConfig.from_env()

    per_role = {
        role: getattr(args, f"{role}_model")
        for role in ("attacker", "target", "evaluator", "judge", "reflector")
        if getattr(args, f"{role}_model")
    }
    if args.model or per_role or args.base_url or args.api_key_env:
        cfg = cfg.with_model_overrides(
            all_roles=args.model,
            per_role=per_role or None,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
        )

    tap_changes = {
        key: getattr(args, key)
        for key in (
            "depth",
            "branching_factor",
            "width",
            "n_streams",
            "keep_last_n",
            "max_attack_attempts",
            "early_stop_score",
            "seed",
        )
        if getattr(args, key) is not None
    }
    if tap_changes:
        cfg = replace(cfg, tap=replace(cfg.tap, **tap_changes))

    if args.max_metric_calls is not None:
        cfg = replace(cfg, gepa=replace(cfg.gepa, max_metric_calls=args.max_metric_calls))

    data_changes: dict[str, Any] = {}
    if args.task_ids is not None:
        data_changes["task_ids"] = _int_tuple(args.task_ids)
    if args.split is not None:
        data_changes["split"] = args.split
        if args.task_ids is None:
            data_changes["task_ids"] = ()
    if args.categories is not None:
        data_changes["categories"] = _str_tuple(args.categories)
    if args.limit is not None:
        data_changes["limit"] = args.limit
    if data_changes:
        cfg = replace(cfg, data=replace(cfg.data, **data_changes))

    if args.runs_dir:
        cfg = replace(cfg, runs_dir=args.runs_dir)
    if args.prompts_dir:
        cfg = replace(cfg, prompts_dir=args.prompts_dir)
    if args.mock:
        cfg = replace(cfg, mock=True)
    return cfg


def _resolve_attacker_prompt(cfg: RunConfig, version: str | None) -> tuple[str, str]:
    """Return (prompt_text, version_id), registering the seed if needed."""
    store = PromptStore(cfg.prompts_dir, role="attacker")
    if version:
        return store.read_text(version), version
    seed = seed_attacker_system_prompt()
    record = store.register(
        seed, source="verbatim:RICommunity/TAP get_attacker_system_prompt"
    )
    return seed, record.version_id


def _data_for_split(cfg: RunConfig, split: str) -> list:
    return select_tasks(replace(cfg.data, split=split, task_ids=()))


def cmd_tap_run(args: argparse.Namespace) -> int:
    cfg = _config_from_args(args)
    tasks = select_tasks(cfg.data)
    prompt_text, version_id = _resolve_attacker_prompt(cfg, args.prompt_version)

    run_id = new_run_id("tap")
    sink = JSONLSink(Path(cfg.runs_dir) / run_id)
    sink.write_json("config.json", {**cfg.to_dict(), "prompt_version": version_id})

    client = build_client(cfg.mock)
    judge = Judge(client=client, model=cfg.model("judge"))

    results = []
    for task in tasks:
        tap_result = run_tap(
            system_prompt=prompt_text,
            goal=task.goal,
            target_str=task.target_str,
            config=cfg.tap,
            client=client,
            models=cfg.models,
            trace_sink=sink,
            run_context={"task_id": task.task_id, "goal": task.goal},
        )
        judged = judge.score(
            goal=task.goal,
            target_str=task.target_str,
            adv_prompt=tap_result.best_payload or "(no valid adversarial prompt)",
            target_response=tap_result.best_target_response or "(no response)",
            attacker_raw=tap_result.best_payload or "(none)",
        )
        sink.emit("judge", {"task_id": task.task_id, **judged.to_dict()})
        results.append(
            {
                "task_id": task.task_id,
                "goal": task.goal,
                "tap_best_score": tap_result.best_score,
                "jailbroken": tap_result.jailbroken,
                "judge_primary": judged.primary_score,
                "stats": tap_result.stats.to_dict(),
            }
        )

    summary = {
        "run_id": run_id,
        "command": "tap-run",
        "prompt_version": version_id,
        "results": results,
        "usage": client.usage.to_dict(),
    }
    sink.write_json("summary.json", summary)
    sink.close()
    print(json.dumps(summary, indent=2))
    return 0


def cmd_gepa_run(args: argparse.Namespace) -> int:
    cfg = _config_from_args(args)
    trainset = select_tasks(cfg.data)
    valset = _data_for_split(cfg, "test") if not cfg.data.task_ids else trainset
    if not valset:
        valset = trainset

    run_id = new_run_id("gepa")
    run_dir = Path(cfg.runs_dir) / run_id
    sink = JSONLSink(run_dir)
    sink.write_json("config.json", cfg.to_dict())

    client = build_client(cfg.mock)
    judge = Judge(client=client, model=cfg.model("judge"))
    adapter = TAPGEPAAdapter(
        client=client,
        models=cfg.models,
        tap_config=cfg.tap,
        judge=judge,
        trace_sink=sink,
    )

    store = PromptStore(cfg.prompts_dir, role="attacker")
    seed_text, seed_version = _resolve_attacker_prompt(cfg, None)

    result = gepa.optimize(
        seed_candidate={COMPONENT: seed_text},
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=make_reflection_lm(client, cfg.model("reflector")),
        reflection_prompt_template=reflection_prompt_template(),
        reflection_minibatch_size=cfg.gepa.reflection_minibatch_size,
        max_metric_calls=cfg.gepa.max_metric_calls,
        use_merge=cfg.gepa.use_merge,
        skip_perfect_score=cfg.gepa.skip_perfect_score,
        seed=cfg.gepa.seed,
        display_progress_bar=False,
    )

    best_text = result.best_candidate[COMPONENT]
    best_version = store.register(best_text, source=f"gepa:{run_id}", parent=seed_version)
    val_scores = getattr(result, "val_aggregate_scores", None)
    summary = {
        "run_id": run_id,
        "command": "gepa-run",
        "seed_version": seed_version,
        "best_version": best_version.version_id,
        "best_idx": result.best_idx,
        "num_candidates": result.num_candidates,
        "total_metric_calls": result.total_metric_calls,
        "val_aggregate_scores": val_scores,
        "usage": client.usage.to_dict(),
    }
    sink.emit("gepa", summary)
    sink.write_json("summary.json", summary)
    sink.close()
    print(json.dumps(summary, indent=2))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    cfg = _config_from_args(args)
    tasks = select_tasks(cfg.data)
    prompt_text, version_id = _resolve_attacker_prompt(cfg, args.prompt_version)

    run_id = new_run_id("eval")
    sink = JSONLSink(Path(cfg.runs_dir) / run_id)
    sink.write_json("config.json", {**cfg.to_dict(), "prompt_version": version_id})

    client = build_client(cfg.mock)
    judge = Judge(client=client, model=cfg.model("judge"))
    adapter = TAPGEPAAdapter(
        client=client,
        models=cfg.models,
        tap_config=cfg.tap,
        judge=judge,
        trace_sink=sink,
    )
    batch = adapter.evaluate(tasks, {COMPONENT: prompt_text}, capture_traces=True)
    mean_score = statistics.fmean(batch.scores) if batch.scores else 0.0
    summary = {
        "run_id": run_id,
        "command": "eval",
        "prompt_version": version_id,
        "split": cfg.data.split if not cfg.data.task_ids else "task_ids",
        "num_tasks": len(tasks),
        "mean_primary_score": mean_score,
        "per_task": [
            {"task_id": task.task_id, "score": score}
            for task, score in zip(tasks, batch.scores, strict=True)
        ],
        "usage": client.usage.to_dict(),
    }
    sink.write_json("summary.json", summary)
    sink.close()
    print(json.dumps(summary, indent=2))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        print(f"no summary.json in {run_dir}", file=sys.stderr)
        return 1
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    lines = [f"# Run report: {summary.get('run_id', run_dir.name)}", ""]
    lines.append(f"- command: {summary.get('command')}")
    for key in ("prompt_version", "best_version", "mean_primary_score", "split"):
        if key in summary:
            lines.append(f"- {key}: {summary[key]}")
    if "results" in summary:
        lines.append("")
        lines.append("| task | tap_best | jailbroken | judge |")
        lines.append("| --- | --- | --- | --- |")
        for row in summary["results"]:
            lines.append(
                f"| {row['task_id']} | {row['tap_best_score']} | "
                f"{row['jailbroken']} | {row['judge_primary']:.2f} |"
            )
    if "usage" in summary:
        usage = summary["usage"]
        lines.append("")
        lines.append(
            f"- queries: {usage.get('calls')} | tokens: {usage.get('total_tokens')}"
        )
    report = "\n".join(lines)
    print(report)
    (run_dir / "report.md").write_text(report + "\n", encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gys", description="TAP + GEPA harness")
    sub = parser.add_subparsers(dest="command", required=True)

    p_tap = sub.add_parser("tap-run", help="Run TAP once per selected task.")
    _add_common_flags(p_tap)
    p_tap.add_argument("--prompt-version", help="Attacker prompt version id.")
    p_tap.set_defaults(func=cmd_tap_run)

    p_gepa = sub.add_parser("gepa-run", help="Optimize the attacker prompt with GEPA.")
    _add_common_flags(p_gepa)
    p_gepa.set_defaults(func=cmd_gepa_run)

    p_eval = sub.add_parser("eval", help="Evaluate a prompt version on a split.")
    _add_common_flags(p_eval)
    p_eval.add_argument("--prompt-version", help="Attacker prompt version id.")
    p_eval.set_defaults(func=cmd_eval)

    p_report = sub.add_parser("report", help="Render a run's summary as Markdown.")
    p_report.add_argument("run_dir", help="Path to runs/<run_id>.")
    p_report.set_defaults(func=cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
