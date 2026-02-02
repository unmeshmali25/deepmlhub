# DVC Configuration - Root-Level Setup

## Summary

Successfully resolved DVC path and directory context issues by implementing a **root-level DVC configuration** with subdirectory project isolation.

## What Was Done

### Problem
- DVC initialized with `--subdir` in project directory created conflicts
- Running `dvc repro` from project directory caused root-level `.dvc/` interference
- DVC couldn't resolve which repository context to use

### Solution: Root-Level DVC with `wdir`

Created a root-level `dvc.yaml` that uses the `wdir` (working directory) directive to execute commands from the project subdirectory while maintaining all DVC tracking at the repository root.

### Files Changed

**Created at Root:**
- `/dvc.yaml` - Main pipeline configuration with `wdir: projects/synth_tabular_classification`
- `/dvc.lock` - Pipeline state tracking (auto-generated)

**Removed from Project:**
- `projects/synth_tabular_classification/dvc.yaml` (moved to root)
- `projects/synth_tabular_classification/dvc.lock` (moved to root)

**Kept in Project:**
- `projects/synth_tabular_classification/params.yaml` - Project-specific configuration
- `projects/synth_tabular_classification/.gitignore` - Git ignore rules

**Updated:**
- `.dvcignore` - Added comprehensive ignore patterns for cache, Python, IDE files
- `.dvc/config` - Already initialized at root level

## Pipeline Stages

The DVC pipeline has 4 stages:

1. **generate** - Creates synthetic data
   - Cmd: `python -m src.data.generate`
   - Output: `data/raw/synthetic_data.csv`

2. **preprocess** - Splits and prepares data
   - Cmd: `python -m src.data.preprocess`
   - Outputs: `data/processed/train.csv`, `data/processed/test.csv`

3. **train** - Trains RandomForest model
   - Cmd: `python -m src.model.train`
   - Outputs: `models/model.joblib`, `metrics/metrics.json`

4. **evaluate** - Evaluates model performance
   - Cmd: `python -m src.model.evaluate`
   - Outputs: `metrics/classification_report.txt`, `metrics/confusion_matrix.txt`

## How It Works

The `wdir` directive in `dvc.yaml`:
```yaml
stages:
  generate:
    cmd: python -m src.data.generate
    wdir: projects/synth_tabular_classification  # ← Key directive
```

This tells DVC to:
1. Run from repository root
2. Change to the specified working directory before executing commands
3. Track all paths relative to the working directory
4. Store lock file and state at repository root

## Benefits

1. **Single DVC Context** - No competing `.dvc/` directories
2. **Multi-Project Ready** - Can add more projects to the root `dvc.yaml` or create separate ones
3. **Clean Structure** - Project code stays isolated, DVC management at root
4. **Git Integration** - All DVC state tracked in one place

## Usage

All DVC commands are run from repository root:

```bash
# View pipeline
dvc dag

# Check status
dvc status

# Run full pipeline
dvc repro

# Run specific stage
dvc repro --single-item generate

# View DAG for specific file
dvc dag dvc.yaml
```

## Other Projects

This repo also has DVC pipelines in:
- `projects/pytorch_00_tensor_gpu_basics/dvc.yaml`
- `projects/pytorch_01_neural_networks/dvc.yaml`

These can remain as separate DVC files or be consolidated into the root config later.

## Git Commands to Complete Setup

```bash
# Stage new files
git add dvc.yaml dvc.lock .dvcignore
git add projects/synth_tabular_classification/dvc.yaml projects/synth_tabular_classification/dvc.lock

# Remove old project-level files from git (but keep in filesystem if needed)
git rm --cached projects/synth_tabular_classification/dvc.yaml

git rm --cached projects/synth_tabular_classification/dvc.lock

# Commit
git commit -m "refactor: Move DVC configuration to root level with wdir

- Consolidate DVC management at repository root
- Use wdir directive for subdirectory project execution
- Fix path conflicts between root and project .dvc directories
- Pipeline tested and working: generate → preprocess → train → evaluate"
```

## Verification

Run this to verify everything works:

```bash
# Activate environment
source .venv/bin/activate

# Check pipeline status
dvc status

# View dependency graph
dvc dag

# Run full pipeline (if needed)
dvc repro
```

## Future Additions

To add DVC to another project:

1. Create pipeline stages in root `dvc.yaml` with new `wdir`:
```yaml
stages:
  new_project_stage:
    cmd: python script.py
    wdir: projects/new_project
```

2. Or create a separate `dvc.yaml` in the project directory (DVC supports multiple)

3. Run from root: `dvc repro` will discover all pipelines
