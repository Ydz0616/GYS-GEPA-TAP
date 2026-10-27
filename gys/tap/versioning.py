"""System-prompt version control.

Every system prompt (the verbatim seed plus each GEPA-proposed candidate) is
stored as a file under ``prompts/attacker/`` and tracked in a JSON manifest that
records content hash, parent lineage, and provenance. This satisfies the
project requirement of version control for system prompts and lets a run be
traced back to the exact prompt text it used.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PromptVersion:
    version_id: str
    role: str
    index: int
    sha256: str
    path: str
    parent: str | None
    source: str
    created_at: str

    def to_dict(self) -> dict:
        return asdict(self)


class PromptStore:
    """File-backed store of versioned system prompts with a JSON manifest."""

    def __init__(self, root: str | Path, role: str = "attacker") -> None:
        self.root = Path(root)
        self.role = role
        self.dir = self.root / role
        self.manifest_path = self.dir / "manifest.json"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._versions: list[PromptVersion] = []
        self._load()

    def _load(self) -> None:
        if self.manifest_path.exists():
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            self._versions = [PromptVersion(**item) for item in data.get("versions", [])]

    def _save(self) -> None:
        payload = {"role": self.role, "versions": [v.to_dict() for v in self._versions]}
        self.manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def __len__(self) -> int:
        return len(self._versions)

    @property
    def versions(self) -> list[PromptVersion]:
        return list(self._versions)

    def latest(self) -> PromptVersion | None:
        return self._versions[-1] if self._versions else None

    def get(self, version_id: str) -> PromptVersion:
        for version in self._versions:
            if version.version_id == version_id:
                return version
        raise KeyError(version_id)

    def read_text(self, version_id: str) -> str:
        return (self.root / self.get(version_id).path).read_text(encoding="utf-8")

    def register(self, text: str, source: str, parent: str | None = None) -> PromptVersion:
        """Register ``text`` as the next version and persist it; dedupes by hash.

        If the text already exists as a version, that existing version is
        returned instead of creating a duplicate file.
        """
        digest = sha256_text(text)
        for version in self._versions:
            if version.sha256 == digest:
                return version
        index = len(self._versions)
        version_id = f"{self.role}_v{index:03d}"
        filename = f"{version_id}.md"
        (self.dir / filename).write_text(text, encoding="utf-8")
        version = PromptVersion(
            version_id=version_id,
            role=self.role,
            index=index,
            sha256=digest,
            path=str(Path(self.role) / filename),
            parent=parent,
            source=source,
            created_at=datetime.now(UTC).isoformat(),
        )
        self._versions.append(version)
        self._save()
        return version
