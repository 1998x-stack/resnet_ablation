"""Offline link checks for the actively maintained repository documentation.

Archived long-form planning documents are intentionally excluded because their
external/historical links may no longer describe the live codebase.
"""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[1]
CURRENT_DOCS = (
    "README.md",
    "AGENTS.md",
    "docs/README.md",
    "configs/README.md",
    "docs/archive/README.md",
    "docs/REVIEW_2026-09-21.md",
    "ChatGPT-ResNet 项目构建.md",
)
LINK = re.compile(r"(?<!!)\[[^]]+\]\(([^)\n]+)\)")


@pytest.mark.parametrize("document", CURRENT_DOCS)
def test_relative_document_links_exist(document: str) -> None:
    path = ROOT / document
    assert path.is_file(), f"missing documented entry point: {document}"
    text = path.read_text(encoding="utf8")
    for target in LINK.findall(text):
        # Skip remote URLs, mailto, and local anchors; decode GitHub-safe UTF-8 paths.
        parsed = urlsplit(target.strip().split(" ", 1)[0])
        if parsed.scheme or parsed.netloc or target.startswith("#"):
            continue
        assert not parsed.path.startswith("/"), f"unexpected absolute path in {document}: {target}"
        if not parsed.path:
            continue
        resolved = path.parent / unquote(parsed.path)
        assert resolved.exists(), f"{document} -> {target}: missing {resolved}"


def test_archives_preserve_source_documents() -> None:
    historical = ROOT / "docs/archive/ChatGPT-ResNet 项目构建.md"
    dated_review = ROOT / "docs/archive/REVIEW_2026-09-21.md"
    assert historical.is_file() and historical.stat().st_size > 50_000
    assert dated_review.is_file() and dated_review.stat().st_size > 5_000


def test_live_entry_points_link_agent_policy_and_runbook() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf8")
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf8")
    assert "[AGENTS.md](AGENTS.md)" in readme
    assert "[运行手册](docs/paper/03_RUNBOOK.md)" in readme
    assert "No leakage" in agents
    assert "tests/test_paper_e2e.py" in agents


def test_gitignore_does_not_hide_new_model_source_files() -> None:
    """A plain 'models/' ignore pattern would hide future source modules."""
    import shutil
    import subprocess

    patterns = (ROOT / ".gitignore").read_text(encoding="utf8").splitlines()
    assert "models/" not in patterns
    assert "/models/" in patterns
    assert "/reports/" in patterns

    if shutil.which("git"):
        check = subprocess.run(
            ["git", "-C", str(ROOT), "check-ignore", "-q", "resnet_ablation/models/new_block.py"],
            check=False,
        )
        assert check.returncode == 1, "new model source files must remain addable to git"
