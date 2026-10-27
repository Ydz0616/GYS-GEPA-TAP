"""The TAP tree loop: branch, prune on-topic, query target, prune by judge score.

This is a faithful but lightweight reimplementation of RICommunity/TAP
(arXiv:2312.02119). Loop semantics match upstream: both pruning phases, the
width cap, conversation truncation, early stop at the configured score, and the
feedback turn fed back to the attacker. GEPA only ever changes the attacker
system prompt - never anything in this file.
"""

from __future__ import annotations

import copy
import random
import string
from dataclasses import asdict, dataclass, field
from typing import Any

from gys.config import ModelConfig, TAPConfig
from gys.llm import LLMClient, Message
from gys.tap.parsing import extract_attacker_json, parse_judge_score, parse_on_topic
from gys.tap.prompts import (
    attacker_system_prompt,
    init_message,
    process_target_response,
    tap_judge_system_prompt,
    tap_on_topic_system_prompt,
)
from gys.tracing import NullSink, TraceSink


def _random_id(rng: random.Random, n: int = 12) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(rng.choice(alphabet) for _ in range(n))


@dataclass
class TAPNode:
    """One candidate attack in the tree, with its full lineage and scores."""

    node_id: str
    parent_id: str
    depth: int
    attacker_raw: str | None = None
    improvement: str | None = None
    adv_prompt: str | None = None
    on_topic: int | None = None
    target_response: str | None = None
    judge_score: int | None = None
    status: str = "generated"  # generated | parse_failed | pruned_phase1 | pruned_phase2 | kept

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TAPStats:
    """Branch and query statistics, including depth at first success."""

    depth_reached: int = 0
    depth_at_success: int | None = None
    branches_generated: int = 0
    branches_parse_failed: int = 0
    pruned_phase1: int = 0
    pruned_phase2: int = 0
    attacker_queries: int = 0
    target_queries: int = 0
    evaluator_queries: int = 0

    @property
    def total_queries(self) -> int:
        return self.attacker_queries + self.target_queries + self.evaluator_queries

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "total_queries": self.total_queries}


@dataclass
class TAPResult:
    """What ``run_tap`` hands back to GEPA (or to the CLI)."""

    best_payload: str | None
    best_score: int
    best_target_response: str | None
    jailbroken: bool
    stats: TAPStats
    trace: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_payload": self.best_payload,
            "best_score": self.best_score,
            "best_target_response": self.best_target_response,
            "jailbroken": self.jailbroken,
            "stats": self.stats.to_dict(),
            "usage": self.usage,
            "trace": self.trace,
        }


@dataclass
class _Branch:
    """A surviving conversation plus the feedback turn queued for the attacker."""

    node: TAPNode
    messages: list[Message]
    pending_feedback: str


def _prune(
    branches: list[_Branch],
    scores: list[int],
    width: int,
    rng: random.Random,
) -> tuple[list[_Branch], list[_Branch]]:
    """Keep the top-``width`` branches with a positive score.

    Mirrors upstream: ties are broken randomly, branches scoring 0 are dropped,
    and if nothing has a positive score the two best-ranked branches are kept so
    the tree does not die out.
    """
    ranked = [(score, index) for index, score in enumerate(scores)]
    rng.shuffle(ranked)
    ranked.sort(key=lambda pair: pair[0], reverse=True)

    keep_count = min(width, len(branches))
    kept_indices = [index for score, index in ranked[:keep_count] if score > 0]
    if not kept_indices:
        kept_indices = [index for _, index in ranked[: min(2, len(ranked))]]

    kept_set = set(kept_indices)
    kept = [branches[i] for i in kept_indices]
    dropped = [branches[i] for i in range(len(branches)) if i not in kept_set]
    return kept, dropped


def run_tap(
    system_prompt: str,
    goal: str,
    target_str: str,
    config: TAPConfig,
    client: LLMClient,
    models: dict[str, ModelConfig],
    trace_sink: TraceSink | None = None,
    run_context: dict[str, Any] | None = None,
) -> TAPResult:
    """Run one TAP attack and return the best payload, its score, stats, and trace.

    ``system_prompt`` is the attacker system prompt template (it may contain
    ``{goal}`` / ``{target_str}`` placeholders). This is the only thing GEPA
    optimizes.
    """
    sink = trace_sink or NullSink()
    context = run_context or {}
    rng = random.Random(config.seed)

    attacker_cfg = models["attacker"]
    target_cfg = models["target"]
    evaluator_cfg = models["evaluator"]

    rendered_system = attacker_system_prompt(system_prompt, goal, target_str)
    stats = TAPStats()
    trace: list[TAPNode] = []

    best_payload: str | None = None
    best_score = 0
    best_response: str | None = None

    branches = [
        _Branch(
            node=TAPNode(node_id=_random_id(rng), parent_id="root", depth=0),
            messages=[{"role": "system", "content": rendered_system}],
            pending_feedback=init_message(goal, target_str),
        )
        for _ in range(config.n_streams)
    ]

    for depth in range(1, config.depth + 1):
        stats.depth_reached = depth

        # ---------------- BRANCH ----------------
        children: list[_Branch] = []
        for _ in range(config.branching_factor):
            for parent in branches:
                node = TAPNode(
                    node_id=_random_id(rng),
                    parent_id=parent.node.node_id,
                    depth=depth,
                )
                messages = copy.deepcopy(parent.messages)
                messages.append({"role": "user", "content": parent.pending_feedback})
                child = _Branch(node=node, messages=messages, pending_feedback="")
                stats.branches_generated += 1

                parsed = None
                raw = ""
                for _attempt in range(config.max_attack_attempts):
                    raw = client.chat(messages, attacker_cfg)
                    stats.attacker_queries += 1
                    parsed = extract_attacker_json(raw)
                    if parsed is not None:
                        break
                node.attacker_raw = raw
                if parsed is None:
                    node.status = "parse_failed"
                    stats.branches_parse_failed += 1
                    trace.append(node)
                    sink.emit("tap", {**context, "event": "branch", **node.to_dict()})
                    continue

                node.improvement = str(parsed["improvement"])
                node.adv_prompt = str(parsed["prompt"])
                messages.append({"role": "assistant", "content": raw})
                children.append(child)

        if not children:
            break

        # ------------- PRUNE: PHASE 1 (on-topic) -------------
        on_topic_system = tap_on_topic_system_prompt(goal)
        on_topic_scores: list[int] = []
        for child in children:
            raw = client.chat(
                [
                    {"role": "system", "content": on_topic_system},
                    {"role": "user", "content": str(child.node.adv_prompt)},
                ],
                evaluator_cfg,
            )
            stats.evaluator_queries += 1
            score = parse_on_topic(raw)
            child.node.on_topic = score
            on_topic_scores.append(score)

        children, dropped = _prune(children, on_topic_scores, config.width, rng)
        stats.pruned_phase1 += len(dropped)
        for branch in dropped:
            branch.node.status = "pruned_phase1"
            trace.append(branch.node)
            sink.emit("tap", {**context, "event": "prune_phase1", **branch.node.to_dict()})
        if not children:
            break

        # ------------- QUERY TARGET AND JUDGE -------------
        judge_system = tap_judge_system_prompt(goal, target_str)
        judge_scores: list[int] = []
        for child in children:
            response = client.chat(
                [{"role": "user", "content": str(child.node.adv_prompt)}],
                target_cfg,
            )
            stats.target_queries += 1
            child.node.target_response = response

            judge_raw = client.chat(
                [
                    {"role": "system", "content": judge_system},
                    {
                        "role": "user",
                        "content": (
                            f"[PROMPT]:{child.node.adv_prompt}\n"
                            f"[ASSISTANT'S RESPONSE]: {response}"
                        ),
                    },
                ],
                evaluator_cfg,
            )
            stats.evaluator_queries += 1
            score = parse_judge_score(judge_raw)
            child.node.judge_score = score
            judge_scores.append(score)

            if score > best_score:
                best_score = score
                best_payload = child.node.adv_prompt
                best_response = response

        # ------------- PRUNE: PHASE 2 (judge score) -------------
        children, dropped = _prune(children, judge_scores, config.width, rng)
        stats.pruned_phase2 += len(dropped)
        for branch in dropped:
            branch.node.status = "pruned_phase2"
            trace.append(branch.node)
            sink.emit("tap", {**context, "event": "prune_phase2", **branch.node.to_dict()})

        for child in children:
            child.node.status = "kept"
            trace.append(child.node)
            sink.emit("tap", {**context, "event": "kept", **child.node.to_dict()})

        # ------------- EARLY STOP -------------
        if any(score >= config.early_stop_score for score in judge_scores):
            stats.depth_at_success = depth
            break

        # Truncate history to the last N exchanges, keeping the system message.
        for child in children:
            system_message = child.messages[0]
            tail = child.messages[1:][-2 * config.keep_last_n :]
            child.messages = [system_message, *tail]
            child.pending_feedback = process_target_response(
                str(child.node.target_response), int(child.node.judge_score or 1), goal
            )

        branches = children

    result = TAPResult(
        best_payload=best_payload,
        best_score=best_score,
        best_target_response=best_response,
        jailbroken=best_score >= config.early_stop_score,
        stats=stats,
        trace=[node.to_dict() for node in trace],
        usage=client.usage.to_dict(),
    )
    sink.emit(
        "tap",
        {
            **context,
            "event": "result",
            "best_score": result.best_score,
            "jailbroken": result.jailbroken,
            "stats": stats.to_dict(),
        },
    )
    return result
