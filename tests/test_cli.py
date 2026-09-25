from pathlib import Path

import pytest

from student_agent.cli import _publish_run


def test_publish_run_replaces_both_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    staging = tmp_path / "staging"
    for base, content in ((root, "old"), (staging, "new")):
        (base / "outputs").mkdir(parents=True)
        (base / "traces").mkdir()
        (base / "outputs" / "case.json").write_text(content, encoding="utf-8")
        (base / "traces" / "trace.jsonl").write_text(content, encoding="utf-8")

    _publish_run(root, staging)

    assert (root / "outputs" / "case.json").read_text(encoding="utf-8") == "new"
    assert (root / "traces" / "trace.jsonl").read_text(encoding="utf-8") == "new"
    assert (staging / "previous_outputs" / "case.json").read_text(encoding="utf-8") == "old"


def test_publish_run_restores_previous_artifacts_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    staging = tmp_path / "staging"
    for base, content in ((root, "old"), (staging, "new")):
        (base / "outputs").mkdir(parents=True)
        (base / "traces").mkdir()
        (base / "outputs" / "case.json").write_text(content, encoding="utf-8")
        (base / "traces" / "trace.jsonl").write_text(content, encoding="utf-8")

    original_rename = Path.rename

    def fail_trace_publish(path: Path, target: Path) -> Path:
        if path == staging / "traces" and target == root / "traces":
            raise OSError("simulated publish failure")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_trace_publish)
    with pytest.raises(OSError, match="simulated"):
        _publish_run(root, staging)

    assert (root / "outputs" / "case.json").read_text(encoding="utf-8") == "old"
    assert (root / "traces" / "trace.jsonl").read_text(encoding="utf-8") == "old"
