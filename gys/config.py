"""Configuration dataclasses for every tunable part of the harness.

Nothing in this module reads the network; `RunConfig.resolve` only consults
environment variables so that a run's full configuration can be serialized into
`runs/<run_id>/config.json` and replayed later.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_MODEL = "gemini-3.5-flash"
DEFAULT_API_KEY_ENV = "GEMINI_API_KEY"

#: The five roles that issue model calls. `evaluator` is TAP's in-loop judge and
#: on-topic scorer; `judge` is the separate GEPA-side judge.
ROLES = ("attacker", "target", "evaluator", "judge", "reflector")


@dataclass(frozen=True)
class ModelConfig:
    """How one role talks to a chat-completions endpoint."""

    model: str = DEFAULT_MODEL
    temperature: float = 0.0
    max_tokens: int = 1024
    base_url: str = DEFAULT_BASE_URL
    api_key_env: str = DEFAULT_API_KEY_ENV
    max_retries: int = 3
    timeout_s: float = 120.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_models() -> dict[str, ModelConfig]:
    """Per-role defaults; temperatures follow the TAP reference implementation."""
    return {
        # TAP's ATTACK_TEMP = 1; the attacker must stay creative across branches.
        "attacker": ModelConfig(temperature=1.0, max_tokens=2048),
        # TAP's TARGET_TEMP = 0 and target_max_n_tokens = 150.
        "target": ModelConfig(temperature=0.0, max_tokens=1024),
        # TAP's evaluator is greedy; it only has to emit "[[n]]" or "[[YES]]".
        "evaluator": ModelConfig(temperature=0.0, max_tokens=1024),
        "judge": ModelConfig(temperature=0.0, max_tokens=2048),
        "reflector": ModelConfig(temperature=1.0, max_tokens=4096),
    }


@dataclass(frozen=True)
class TAPConfig:
    """Tree-of-attacks parameters. Defaults are the lightweight course-project
    settings; the reference implementation's defaults are noted alongside."""

    depth: int = 3  # upstream default 10
    branching_factor: int = 2  # upstream default 4
    width: int = 2  # upstream default 10
    n_streams: int = 1
    keep_last_n: int = 3
    max_attack_attempts: int = 5
    early_stop_score: int = 10
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GEPAConfig:
    """GEPA budget and reflection knobs. One metric call is a full TAP run."""

    max_metric_calls: int = 12
    reflection_minibatch_size: int = 1
    seed: int = 0
    use_merge: bool = False
    skip_perfect_score: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DataConfig:
    """AdvBench subset selection and splitting."""

    csv_path: str = "data/advbench_subset.csv"
    split: str = "train"
    train_frac: float = 0.6
    test_frac: float = 0.2
    seed: int = 0
    task_ids: tuple[int, ...] = (1,)
    categories: tuple[str, ...] = ()
    limit: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunConfig:
    """Everything one `tap-run`, `gepa-run`, or `eval` invocation needs."""

    models: dict[str, ModelConfig] = field(default_factory=default_models)
    tap: TAPConfig = field(default_factory=TAPConfig)
    gepa: GEPAConfig = field(default_factory=GEPAConfig)
    data: DataConfig = field(default_factory=DataConfig)
    prompts_dir: str = "prompts"
    runs_dir: str = "runs"
    mock: bool = False

    def __post_init__(self) -> None:
        missing = [role for role in ROLES if role not in self.models]
        if missing:
            raise ValueError(f"missing model configuration for roles: {missing}")

    def model(self, role: str) -> ModelConfig:
        if role not in self.models:
            raise KeyError(f"unknown role {role!r}; known roles: {sorted(self.models)}")
        return self.models[role]

    def with_model_overrides(
        self,
        *,
        all_roles: str | None = None,
        per_role: dict[str, str] | None = None,
        base_url: str | None = None,
        api_key_env: str | None = None,
    ) -> RunConfig:
        """Return a copy with model names (and optionally transport) overridden."""
        models = dict(self.models)
        for role, cfg in models.items():
            changes: dict[str, Any] = {}
            if all_roles:
                changes["model"] = all_roles
            if per_role and per_role.get(role):
                changes["model"] = per_role[role]
            if base_url:
                changes["base_url"] = base_url
            if api_key_env:
                changes["api_key_env"] = api_key_env
            if changes:
                models[role] = replace(cfg, **changes)
        return replace(self, models=models)

    @classmethod
    def from_env(cls) -> RunConfig:
        """Build a config from `GYS_BASE_URL` / `GYS_MODEL` / `GYS_API_KEY_ENV`."""
        cfg = cls()
        return cfg.with_model_overrides(
            all_roles=os.environ.get("GYS_MODEL") or None,
            base_url=os.environ.get("GYS_BASE_URL") or None,
            api_key_env=os.environ.get("GYS_API_KEY_ENV") or None,
        )

    @classmethod
    def from_file(cls, path: str | Path) -> RunConfig:
        """Load a JSON config; unspecified sections keep their defaults."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        base = cls.from_env()
        models = dict(base.models)
        for role, overrides in (raw.get("models") or {}).items():
            if role not in models:
                raise ValueError(f"unknown role {role!r} in {path}")
            models[role] = replace(models[role], **overrides)
        return cls(
            models=models,
            tap=replace(base.tap, **(raw.get("tap") or {})),
            gepa=replace(base.gepa, **(raw.get("gepa") or {})),
            data=cls._data_from_raw(base.data, raw.get("data") or {}),
            prompts_dir=raw.get("prompts_dir", base.prompts_dir),
            runs_dir=raw.get("runs_dir", base.runs_dir),
            mock=bool(raw.get("mock", base.mock)),
        )

    @staticmethod
    def _data_from_raw(base: DataConfig, raw: dict[str, Any]) -> DataConfig:
        raw = dict(raw)
        for key in ("task_ids", "categories"):
            if key in raw and raw[key] is not None:
                raw[key] = tuple(raw[key])
        return replace(base, **raw)

    def to_dict(self) -> dict[str, Any]:
        return {
            "models": {role: cfg.to_dict() for role, cfg in self.models.items()},
            "tap": self.tap.to_dict(),
            "gepa": self.gepa.to_dict(),
            "data": self.data.to_dict(),
            "prompts_dir": self.prompts_dir,
            "runs_dir": self.runs_dir,
            "mock": self.mock,
        }
