import json

from gys.config import ModelConfig, TAPConfig, default_models
from gys.datasets import Task
from gys.gepa_adapter import COMPONENT, TAPGEPAAdapter
from gys.judge import Judge
from gys.llm import LLMClient
from gys.tap.prompts import seed_attacker_system_prompt

JUDGE_JSON = json.dumps(
    {
        "jailbreak_success": 6,
        "goal_fidelity": 7,
        "stealth": 5,
        "strategy_quality": 6,
        "format_compliance": 9,
        "side_info": "Obfuscate the sensitive noun and add an authority frame.",
    }
)


class RoleRoutingBackend:
    """Routes by prompt content so all five roles can share one mock model."""

    def __init__(self, attacker_valid=True):
        self.attacker_valid = attacker_valid

    def complete(self, messages, cfg):
        text = " ".join(m["content"] for m in messages)
        if "red teaming assistant" in text and "FORMAT" in text:
            if not self.attacker_valid:
                return "no json", 1, 1, "stop"
            return json.dumps({"improvement": "i", "prompt": "adv"}), 1, 1, "stop"
        if "asks for the same information" in text:
            return "Response: [[YES]]", 1, 1, "stop"
        if "impartial judge" in text:
            return "Rating: [[7]]", 1, 1, "stop"
        if "professional red-teaming evaluator" in text:
            return JUDGE_JSON, 1, 1, "stop"
        return "target response text", 1, 1, "stop"


TASKS = [
    Task(1, "do the thing", "Sure, here", "bomb", 7),
    Task(2, "do other thing", "Sure, there", "hacking", 1),
]


def _adapter(backend):
    client = LLMClient(backend=backend)
    return TAPGEPAAdapter(
        client=client,
        models=default_models(),
        tap_config=TAPConfig(depth=1, branching_factor=2, width=2, n_streams=1),
        judge=Judge(client=client, model=ModelConfig()),
    )


def test_evaluate_returns_aligned_lengths_and_scores():
    adapter = _adapter(RoleRoutingBackend())
    batch = adapter.evaluate(TASKS, {COMPONENT: seed_attacker_system_prompt()})
    assert len(batch.outputs) == len(TASKS)
    assert len(batch.scores) == len(TASKS)
    assert batch.trajectories is None
    assert all(score == 0.6 for score in batch.scores)
    assert batch.objective_scores is not None
    assert "tap_best_judge" in batch.objective_scores[0]


def test_evaluate_with_capture_traces_populates_trajectories():
    adapter = _adapter(RoleRoutingBackend())
    batch = adapter.evaluate(
        TASKS, {COMPONENT: seed_attacker_system_prompt()}, capture_traces=True
    )
    assert batch.trajectories is not None
    assert len(batch.trajectories) == len(TASKS)
    assert batch.trajectories[0]["best_payload"] == "adv"
    assert batch.trajectories[0]["judge"]["jailbreak_success"] == 6


def test_evaluate_handles_attacker_parse_failure_without_raising():
    adapter = _adapter(RoleRoutingBackend(attacker_valid=False))
    batch = adapter.evaluate(
        TASKS, {COMPONENT: seed_attacker_system_prompt()}, capture_traces=True
    )
    assert len(batch.scores) == len(TASKS)
    assert batch.trajectories[0]["best_payload"] is None


def test_make_reflective_dataset_shape():
    adapter = _adapter(RoleRoutingBackend())
    candidate = {COMPONENT: seed_attacker_system_prompt()}
    batch = adapter.evaluate(TASKS, candidate, capture_traces=True)
    dataset = adapter.make_reflective_dataset(candidate, batch, [COMPONENT])
    assert set(dataset) == {COMPONENT}
    records = dataset[COMPONENT]
    assert len(records) == len(TASKS)
    record = records[0]
    assert set(record) == {"Inputs", "Generated Outputs", "Feedback"}
    assert "Obfuscate the sensitive noun" in record["Feedback"]
    assert record["Inputs"]["goal"] == "do the thing"
    json.dumps(dataset)  # must stay JSON-serializable for GEPA
