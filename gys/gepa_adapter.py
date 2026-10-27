"""GEPAAdapter that optimizes the attacker system prompt against TAP.

GEPA optimizes exactly one component - ``attacker_system_prompt`` - and never
touches the TAP loop. For each task, ``evaluate`` runs TAP with the candidate
prompt, judges the best payload TAP produced, and returns the normalized judge
score. ``make_reflective_dataset`` turns each task's trajectory into an
Inputs/Generated Outputs/Feedback record so GEPA's reflector can rewrite the
prompt from concrete, per-task feedback.
"""

from __future__ import annotations

from typing import Any

from gepa.core.adapter import EvaluationBatch, GEPAAdapter

from gys.config import ModelConfig, TAPConfig
from gys.datasets import Task
from gys.judge import Judge
from gys.llm import LLMClient
from gys.tap.loop import run_tap
from gys.tracing import NullSink, TraceSink

COMPONENT = "attacker_system_prompt"


class TAPGEPAAdapter(GEPAAdapter):
    """Bridges the TAP loop and the GEPA judge into GEPA's evaluate/reflect API."""

    def __init__(
        self,
        client: LLMClient,
        models: dict[str, ModelConfig],
        tap_config: TAPConfig,
        judge: Judge,
        trace_sink: TraceSink | None = None,
    ) -> None:
        self.client = client
        self.models = models
        self.tap_config = tap_config
        self.judge = judge
        self.sink = trace_sink or NullSink()

    def evaluate(
        self,
        batch: list[Task],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        system_prompt = candidate[COMPONENT]
        outputs: list[dict[str, Any]] = []
        scores: list[float] = []
        objective_scores: list[dict[str, float]] = []
        trajectories: list[dict[str, Any]] | None = [] if capture_traces else None

        for task in batch:
            context = {"task_id": task.task_id, "goal": task.goal}
            tap_result = run_tap(
                system_prompt=system_prompt,
                goal=task.goal,
                target_str=task.target_str,
                config=self.tap_config,
                client=self.client,
                models=self.models,
                trace_sink=self.sink,
                run_context=context,
            )

            if tap_result.best_payload is None:
                judged = self.judge.score(
                    goal=task.goal,
                    target_str=task.target_str,
                    adv_prompt="(no valid adversarial prompt was produced)",
                    target_response="(no target response)",
                    attacker_raw="(attacker never emitted valid JSON)",
                )
            else:
                judged = self.judge.score(
                    goal=task.goal,
                    target_str=task.target_str,
                    adv_prompt=tap_result.best_payload,
                    target_response=tap_result.best_target_response or "",
                    attacker_raw=tap_result.best_payload,
                )

            score = judged.primary_score
            scores.append(score)
            objective_scores.append(
                {
                    "tap_best_judge": tap_result.best_score / 10.0,
                    **judged.objective_scores(),
                }
            )
            output = {
                "task_id": task.task_id,
                "best_payload": tap_result.best_payload,
                "best_target_response": tap_result.best_target_response,
                "tap_best_score": tap_result.best_score,
                "jailbroken": tap_result.jailbroken,
                "judge": judged.to_dict(),
                "stats": tap_result.stats.to_dict(),
            }
            outputs.append(output)

            self.sink.emit(
                "judge",
                {
                    "task_id": task.task_id,
                    "goal": task.goal,
                    "score": score,
                    **judged.to_dict(),
                },
            )

            if trajectories is not None:
                trajectories.append(
                    {
                        "task": task.to_dict(),
                        "best_payload": tap_result.best_payload,
                        "best_target_response": tap_result.best_target_response,
                        "tap_best_score": tap_result.best_score,
                        "judge": judged.to_dict(),
                        "stats": tap_result.stats.to_dict(),
                    }
                )

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
            objective_scores=objective_scores,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        records: list[dict[str, Any]] = []
        for trajectory in eval_batch.trajectories or []:
            judge = trajectory["judge"]
            task = trajectory["task"]
            records.append(
                {
                    "Inputs": {
                        "goal": task["goal"],
                        "target_prefix": task["target_str"],
                    },
                    "Generated Outputs": {
                        "best_adversarial_prompt": trajectory["best_payload"] or "(none)",
                        "target_response": trajectory["best_target_response"] or "(none)",
                        "tap_best_judge_score": str(trajectory["tap_best_score"]),
                    },
                    "Feedback": (
                        f"jailbreak_success={judge['jailbreak_success']}/10, "
                        f"goal_fidelity={judge['goal_fidelity']}/10, "
                        f"stealth={judge['stealth']}/10, "
                        f"strategy_quality={judge['strategy_quality']}/10, "
                        f"format_compliance={judge['format_compliance']}/10. "
                        f"{judge['side_info']}"
                    ),
                }
            )
        self.sink.emit("reflector", {"component": COMPONENT, "num_records": len(records)})
        return {component: records for component in components_to_update}
