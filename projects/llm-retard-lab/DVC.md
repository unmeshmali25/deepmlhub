# DVC Runbook

Operational procedures for data and model artifacts in `llm-retard-lab`.

## The split

- **CI never touches bytes.** No `dvc pull` / `dvc repro` / `dvc push` in GitHub
  Actions. CI only validates pointers (`dvc dag`, lock/pointer contract tests)
  and code (ruff, pytest). See `.github/workflows/dvc-smoke.yaml`.
- **Humans own the data.** You and any collaborator run `pull → (change) →
  repro → push` on your own machines. The GCS remote is shared.

**Remote:** `gs://deepmlhub-llm-posttraining-dvc` (tracked in `.dvc/config`).
Credentials are **not** in git; each person configures their own.

## 1. Install

```bash
python -m venv .venv && source .venv/bin/activate   # or reuse repo .venv
pip install -e "projects/llm-retard-lab[dev,train]"
```

If you only need the data tooling, `pip install -e ".[dev]"` from the project
directory is enough (`dev` includes `dvc[gs]`, pinned to match everyone else).

## 2. Authenticate (once per machine)

Use your own Google identity — do not share the service-account key.

```bash
gcloud auth application-default login
```

Access is granted per person:
- `roles/storage.objectViewer` on the bucket → read (`dvc pull`)
- `roles/storage.objectCreator` (or `objectAdmin`) → write (`dvc push`)

Ask the bucket owner to grant these on `deepmlhub-llm-posttraining-dvc`.

## 3. Get the data

```bash
cd projects/llm-retard-lab
dvc pull --dry                                  # preview size before downloading
dvc pull data/01_cold_start_cot_sft/data.jsonl.dvc   # or bare `dvc pull` for all
dvc status                                      # expect: "Data and pipelines are up to date."
```

`--dry` is important once data grows; a bare `dvc pull` fetches every out in the
lock (including large model adapters).

## 4. Add data additively

Never `git add` a `.jsonl`. Raw bytes go to DVC, pointers go to git.

```bash
# 1. edit/append the data file normally
# 2. hand it to DVC (updates the .dvc pointer + stage cache)
dvc add data/01_cold_start_cot_sft/data.jsonl
# 3. upload bytes to GCS
dvc push
# 4. commit the pointer(s), not the bytes
git add data/01_cold_start_cot_sft/data.jsonl.dvc data/01_cold_start_cot_sft/.gitignore
git commit -m "data(llm): <what changed>"
git push
```

## 5. Retrain and publish results

```bash
# from a clean base: make sure you have the latest pointers and bytes
git pull
dvc pull data/01_cold_start_cot_sft/data.jsonl.dvc
# run the pipeline on your machine
dvc repro
dvc metrics show
# publish artifacts + the updated stage hashes
dvc push
git add dvc.lock
git commit -m "train(llm): <what changed>"
git push
```

`dvc.lock` records the hashes CI validates. If you skip committing it, the
contract test `test_pointer_hashes_match_lock_deps` will fail.

## 6. Lock / pointer conflicts

`dvc.lock` and `*.dvc` are small YAML files and will conflict if two people
train on stale bases. Never hand-edit them.

```bash
git pull                 # resolve the conflict by taking either side
dvc pull                 # fetch bytes for the pointers on disk
dvc repro                # regenerate the correct hashes locally
dvc push
git add dvc.lock *.dvc
git commit               # commit the regenerated pointer
```

Whoever repros last owns the final hash. When in doubt, re-run `dvc repro`.

## 7. Never do

- `git add -f` a `.jsonl` (or any DVC-tracked path) — reintroduces double-ownership
  and breaks `dvc repro` everywhere.
- Hand-edit `dvc.lock` / `.dvc` pointers.
- Add `dvc pull` / `dvc repro` / `dvc push` to CI — data will only grow; CI stays
  byte-free by design.
- Share the service-account key file; grant per-person IAM instead.

## Known debt

- Old data blobs remain in git history (pre-DVC migration). Clones carry them
  until a `git filter-repo` is run — only worth doing if the repo goes public.
- The repo root has an empty `.dvc/` directory that is unrelated to this project.
