#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXCLUDED_FILES = {
    Path("CLAUDE.md"),
    Path("学习时间线.md"),
}
EXCLUDED_BASENAMES = {
    "README.md",
}
EXCLUDED_PREFIXES = (
    Path("docs/superpowers/specs"),
)


def iter_markdown_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.md")):
        relative_path = path.relative_to(root)
        if relative_path in EXCLUDED_FILES:
            continue
        if path.name in EXCLUDED_BASENAMES:
            continue
        if any(relative_path.is_relative_to(prefix) for prefix in EXCLUDED_PREFIXES):
            continue
        yield path


def parse_front_matter(text: str) -> Optional[Dict[str, str]]:
    if not text.startswith("---\n"):
        return None

    lines = text.splitlines()
    metadata: Dict[str, str] = {}

    for line in lines[1:]:
        if line == "---":
            return metadata
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()

    return None


def is_unarchived_note(path: Path) -> bool:
    metadata = parse_front_matter(path.read_text(encoding="utf-8"))
    if metadata is None:
        return False
    return metadata.get("archived", "").lower() == "false"


def main() -> None:
    for path in iter_markdown_files(PROJECT_ROOT):
        if is_unarchived_note(path):
            print(path.relative_to(PROJECT_ROOT).as_posix())


if __name__ == "__main__":
    main()
