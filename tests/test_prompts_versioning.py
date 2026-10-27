from gys.tap.prompts import (
    init_message,
    process_target_response,
    render,
    seed_attacker_system_prompt,
    tap_judge_system_prompt,
    tap_on_topic_system_prompt,
)
from gys.tap.versioning import PromptStore, sha256_text


def test_seed_prompt_has_placeholders_and_literal_braces():
    seed = seed_attacker_system_prompt()
    assert "{goal}" in seed
    assert "{target_str}" in seed
    # The JSON examples must survive as literal single braces.
    assert '"improvement":' in seed


def test_render_substitutes_only_named_tokens():
    out = render("a {goal} b {target_str} c {keep}", goal="G", target_str="T")
    assert out == "a G b T c {keep}"


def test_attacker_prompt_renders_without_format_errors():
    rendered = render(seed_attacker_system_prompt(), goal="G", target_str="T")
    assert "{goal}" not in rendered
    assert "{target_str}" not in rendered
    assert "G" in rendered and "T" in rendered


def test_tap_evaluator_prompts_render():
    judge = tap_judge_system_prompt("G", "T")
    assert "TASK: `G`" in judge
    assert "Rating: [[rating]]" in judge
    on_topic = tap_on_topic_system_prompt("G")
    assert "TASK: `G`" in on_topic
    assert "[[response]]" in on_topic


def test_runtime_messages():
    assert init_message("G", "T").endswith("Begin.")
    assert "'G'" in init_message("G", "T")
    feedback = process_target_response("OUT", 5, "G")
    assert "LANGUAGE MODEL OUTPUT: OUT" in feedback
    assert "SCORE: 5" in feedback


def test_prompt_store_registers_versions_with_lineage(tmp_path):
    store = PromptStore(tmp_path, role="attacker")
    v0 = store.register("seed text", source="seed")
    v1 = store.register("child text", source="gepa", parent=v0.version_id)
    assert v0.version_id == "attacker_v000"
    assert v1.version_id == "attacker_v001"
    assert v1.parent == "attacker_v000"
    assert v0.sha256 == sha256_text("seed text")
    assert store.read_text(v1.version_id) == "child text"
    assert store.latest().version_id == v1.version_id


def test_prompt_store_dedupes_by_hash(tmp_path):
    store = PromptStore(tmp_path, role="attacker")
    a = store.register("same", source="seed")
    b = store.register("same", source="gepa")
    assert a.version_id == b.version_id
    assert len(store) == 1


def test_prompt_store_reloads_manifest(tmp_path):
    store = PromptStore(tmp_path, role="attacker")
    store.register("seed text", source="seed")
    reopened = PromptStore(tmp_path, role="attacker")
    assert len(reopened) == 1
    assert reopened.latest().version_id == "attacker_v000"
