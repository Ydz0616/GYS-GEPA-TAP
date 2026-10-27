import json

from gys.config import TAPConfig, default_models
from gys.llm import LLMClient
from gys.tap.loop import run_tap
from gys.tap.prompts import seed_attacker_system_prompt
from gys.tracing import JSONLSink


class ScriptedBackend:
    """Content-aware fake: routes by the system prompt so every role is distinct.

    The attacker always emits valid JSON; the target echoes a canned response;
    on-topic always says YES; the judge returns ``judge_score`` (optionally a
    callable of the attacker-query count so a test can trigger early stop).
    """

    def __init__(self, judge_score=3, attacker_valid=True):
        self.judge_score = judge_score
        self.attacker_valid = attacker_valid
        self.attacker_calls = 0

    def complete(self, messages, cfg):
        system = messages[0]["content"] if messages[0]["role"] == "system" else ""
        if "red teaming assistant" in system and "FORMAT" in system:
            self.attacker_calls += 1
            if self.attacker_valid:
                content = json.dumps({"improvement": "x", "prompt": "adv prompt"})
            else:
                content = "not json at all"
            return content, 5, 5, "stop"
        if "asks for the same information" in system:
            return "Response: [[YES]]", 1, 1, "stop"
        if "impartial judge" in system:
            score = self.judge_score
            if callable(score):
                score = score(self.attacker_calls)
            return f"Rating: [[{score}]]", 1, 1, "stop"
        return "here is the target response", 2, 2, "stop"


def _run(backend, cfg=None, sink=None):
    client = LLMClient(backend=backend)
    return run_tap(
        system_prompt=seed_attacker_system_prompt(),
        goal="do the thing",
        target_str="Sure, here",
        config=cfg or TAPConfig(depth=2, branching_factor=2, width=2, n_streams=1),
        client=client,
        models=default_models(),
        trace_sink=sink,
    )


def test_basic_run_scores_and_stats():
    result = _run(ScriptedBackend(judge_score=4))
    assert result.best_score == 4
    assert result.best_payload == "adv prompt"
    assert result.jailbroken is False
    assert result.stats.branches_generated > 0
    assert result.stats.target_queries > 0
    assert result.stats.total_queries == (
        result.stats.attacker_queries
        + result.stats.target_queries
        + result.stats.evaluator_queries
    )
    assert len(result.trace) > 0


def test_early_stop_on_full_score():
    cfg = TAPConfig(depth=5, branching_factor=2, width=2, n_streams=1)
    result = _run(ScriptedBackend(judge_score=10), cfg=cfg)
    assert result.jailbroken is True
    assert result.best_score == 10
    assert result.stats.depth_at_success == 1
    assert result.stats.depth_reached == 1


def test_width_cap_limits_kept_branches():
    # 3 branches generated per depth, width 1 => at most 1 kept per phase.
    cfg = TAPConfig(depth=1, branching_factor=3, width=1, n_streams=1)
    result = _run(ScriptedBackend(judge_score=5), cfg=cfg)
    kept = [n for n in result.trace if n["status"] == "kept"]
    assert len(kept) <= 1
    assert result.stats.pruned_phase2 >= 0


def test_parse_failure_is_recorded_and_not_fatal():
    cfg = TAPConfig(
        depth=1, branching_factor=2, width=2, n_streams=1, max_attack_attempts=2
    )
    result = _run(ScriptedBackend(attacker_valid=False), cfg=cfg)
    assert result.stats.branches_parse_failed == result.stats.branches_generated
    assert result.best_payload is None
    failed = [n for n in result.trace if n["status"] == "parse_failed"]
    assert len(failed) == result.stats.branches_generated


def test_trace_sink_writes_jsonl(tmp_path):
    sink = JSONLSink(tmp_path / "run1")
    _run(ScriptedBackend(judge_score=6), sink=sink)
    sink.close()
    lines = (tmp_path / "run1" / "tap.jsonl").read_text().splitlines()
    assert len(lines) > 0
    events = {json.loads(line)["event"] for line in lines}
    assert "result" in events
    assert "kept" in events
