import json
import shutil
from pathlib import Path

from gys.cli import main


def _prompts(tmp_path):
    dst = tmp_path / "prompts"
    shutil.copytree("prompts", dst)
    return str(dst)


def _only_run(runs_dir: Path) -> Path:
    runs = list(runs_dir.iterdir())
    assert len(runs) == 1
    return runs[0]


def test_tap_run_mock_writes_artifacts(tmp_path):
    runs = tmp_path / "runs"
    code = main(
        ["tap-run", "--mock", "--runs-dir", str(runs), "--prompts-dir", _prompts(tmp_path),
         "--task-ids", "1", "--depth", "2"]
    )
    assert code == 0
    run = _only_run(runs)
    for name in ("config.json", "tap.jsonl", "judge.jsonl", "summary.json"):
        assert (run / name).exists(), name
    summary = json.loads((run / "summary.json").read_text())
    assert summary["results"][0]["task_id"] == 1
    assert json.loads((run / "config.json").read_text())["tap"]["depth"] == 2
    assert main(["report", str(run)]) == 0
    assert (run / "report.md").exists()


def test_gepa_run_mock_writes_artifacts_and_versions(tmp_path):
    runs = tmp_path / "runs"
    prompts = _prompts(tmp_path)
    code = main(
        ["gepa-run", "--mock", "--runs-dir", str(runs), "--prompts-dir", prompts,
         "--task-ids", "1", "--depth", "1", "--max-metric-calls", "3"]
    )
    assert code == 0
    run = _only_run(runs)
    for name in ("config.json", "tap.jsonl", "judge.jsonl", "gepa.jsonl", "summary.json"):
        assert (run / name).exists(), name
    summary = json.loads((run / "summary.json").read_text())
    assert summary["seed_version"] == "attacker_v000"
    manifest = json.loads((Path(prompts) / "attacker" / "manifest.json").read_text())
    assert manifest["versions"][0]["version_id"] == "attacker_v000"


def test_eval_mock(tmp_path):
    runs = tmp_path / "runs"
    code = main(
        ["eval", "--mock", "--runs-dir", str(runs), "--prompts-dir", _prompts(tmp_path),
         "--split", "heldout", "--limit", "2", "--depth", "1"]
    )
    assert code == 0
    summary = json.loads((_only_run(runs) / "summary.json").read_text())
    assert summary["num_tasks"] == 2
    assert summary["split"] == "heldout"
