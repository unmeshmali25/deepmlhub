import subprocess
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parents[1]
REPO_ROOT = Path(__file__).parents[3]

DVC_YAML = PROJECT_ROOT / "dvc.yaml"
DVC_LOCK = PROJECT_ROOT / "dvc.lock"

# Empty placeholder files that are fine to keep in git. Real data bytes are not.
GIT_TRACKED_DATA_BYTES_ALLOWLIST = {"failures.jsonl"}


def _load_dvc_yaml():
    return yaml.safe_load(DVC_YAML.read_text())


def _load_dvc_lock():
    return yaml.safe_load(DVC_LOCK.read_text())


def _dvc_pointer_files():
    return sorted(p for p in PROJECT_ROOT.rglob("*.dvc") if p.is_file())


def _git_ls_files(pathspec):
    result = subprocess.run(
        ["git", "ls-files", pathspec],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def test_dvc_yaml_and_lock_define_same_stages():
    yaml_stages = set(_load_dvc_yaml()["stages"])
    lock_stages = set(_load_dvc_lock()["stages"])
    assert yaml_stages == lock_stages


def test_lock_outs_have_md5():
    for stage in _load_dvc_lock()["stages"].values():
        for out in stage.get("outs", []):
            assert out.get("md5"), f"missing md5 for lock out {out.get('path')}"


def test_dvc_pointers_have_hash_and_size():
    pointers = _dvc_pointer_files()
    assert pointers, "no .dvc pointer files found"
    for pointer in pointers:
        outs = yaml.safe_load(pointer.read_text())["outs"]
        assert len(outs) == 1, f"{pointer.name} should have exactly one out"
        assert outs[0].get("md5"), f"{pointer.name} missing md5"
        assert outs[0].get("size", 0) > 0, f"{pointer.name} missing size"


def test_dvc_pointers_are_git_tracked():
    for pointer in _dvc_pointer_files():
        rel = pointer.relative_to(REPO_ROOT).as_posix()
        assert _git_ls_files(rel), f"{rel} is not git-tracked (CI will never see it)"


def test_pointer_hashes_match_lock_deps():
    lock_deps = {}
    for stage in _load_dvc_lock()["stages"].values():
        for dep in stage.get("deps", []):
            lock_deps[dep["path"]] = dep.get("md5")

    for pointer in _dvc_pointer_files():
        data_path = pointer.relative_to(PROJECT_ROOT).with_suffix("").as_posix()
        if data_path not in lock_deps:
            continue  # standalone pointer, not an input to any stage
        pointer_md5 = yaml.safe_load(pointer.read_text())["outs"][0]["md5"]
        assert lock_deps[data_path] == pointer_md5, (
            f"{pointer.name} md5 differs from dvc.lock dep {data_path}; "
            "run `dvc repro` and commit the updated lock"
        )


def test_no_data_bytes_committed_to_git():
    tracked = _git_ls_files("projects/llm-retard-lab/data")
    offenders = [
        path
        for path in tracked
        if path.endswith(".jsonl")
        and Path(path).name not in GIT_TRACKED_DATA_BYTES_ALLOWLIST
    ]
    assert not offenders, (
        f"raw data bytes committed to git (expected DVC pointers instead): {offenders}"
    )
