# Sprint 1: DVC Pipeline Setup

**Status**: ✅ Complete  
**Completion Date**: 2026-02-01  
**Sprint Goal**: Establish data versioning and pipeline reproducibility using DVC

---

## Summary

Set up DVC (Data Version Control) for the synthetic tabular classification project. Created a 4-stage pipeline that tracks data and model artifacts, enabling full reproducibility.

**Results**:
- Pipeline runs successfully from generate → preprocess → train → evaluate
- Training accuracy: 99.25%
- Test accuracy: 90.0%
- dvc.lock file created with hashes for all outputs

---

## Completed Tasks

### [AI] 1.1: Create DVC Pipeline File ✅
**Status**: ✅ Complete (2026-02-01)  
**Time Taken**: 45 minutes  
**Priority**: High

**Delivered**:
- Created `projects/synth_tabular_classification/dvc.yaml`
- 4 stages: generate → preprocess → train → evaluate
- Proper dependency chain established
- metrics/metrics.json tracked as DVC output (not cached in Git)

**Pipeline Definition**:
```yaml
stages:
  generate:
    cmd: python -m src.data.generate
    deps:
      - src/data/generate.py
      - configs/config.yaml
    outs:
      - data/raw/synthetic_data.csv

  preprocess:
    cmd: python -m src.data.preprocess
    deps:
      - src/data/preprocess.py
      - data/raw/synthetic_data.csv
      - configs/config.yaml
    outs:
      - data/processed/train.csv
      - data/processed/test.csv

  train:
    cmd: python -m src.model.train
    deps:
      - src/model/train.py
      - data/processed/train.csv
      - configs/config.yaml
    outs:
      - models/model.joblib
    metrics:
      - metrics/metrics.json:
          cache: false

  evaluate:
    cmd: python -m src.model.evaluate
    deps:
      - src/model/evaluate.py
      - models/model.joblib
      - data/processed/test.csv
    metrics:
      - metrics/metrics.json:
          cache: false
```

**Files Changed**:
- `projects/synth_tabular_classification/dvc.yaml` (new)

**Verification**:
```bash
cd projects/synth_tabular_classification
dvc dag  # Shows 4-stage pipeline graph
```

---

### [AI] 1.2: Create params.yaml for DVC ✅
**Status**: ✅ Complete (2026-02-01)  
**Time Taken**: 10 minutes  
**Priority**: High

**Delivered**:
- Created `projects/synth_tabular_classification/params.yaml`
- 6 parameters: n_samples, n_features, n_classes, test_size, n_estimators, max_depth
- Matches current config.yaml values

**Content**:
```yaml
data:
  n_samples: 1000
  n_features: 10
  n_classes: 2
  test_size: 0.2

model:
  n_estimators: 50
  max_depth: 10
```

**Files Changed**:
- `projects/synth_tabular_classification/params.yaml` (new)

---

### [AI] 1.3: Initialize DVC in Project ✅
**Status**: ✅ Complete (2026-02-01)  
**Time Taken**: 30 minutes  
**Priority**: High

**Delivered**:
- DVC initialized in project subdirectory with `dvc init --subdir`
- `.dvc/` directory created with config
- dvc.lock auto-generated with content hashes
- Pipeline verified with `dvc repro`

**Results**:
- Training accuracy: 99.25%
- Test accuracy: 90.0%
- All 4 stages execute successfully

**Files Changed**:
- `projects/synth_tabular_classification/.dvc/` (new directory)
- `projects/synth_tabular_classification/dvc.lock` (new)
- `projects/synth_tabular_classification/metrics/metrics.json` (switched from Git to DVC tracking)

**Verification Commands**:
```bash
cd projects/synth_tabular_classification
source ../../.venv/bin/activate
dvc dag       # Shows pipeline dependency graph
dvc repro     # Runs full pipeline (all stages up to date)
dvc status    # Shows "Pipeline is up to date"
```

---

## Files Created

### New Files
- `projects/synth_tabular_classification/dvc.yaml` - DVC pipeline definition
- `projects/synth_tabular_classification/params.yaml` - DVC parameters
- `projects/synth_tabular_classification/.dvc/config` - DVC configuration
- `projects/synth_tabular_classification/dvc.lock` - DVC lock file (auto-generated)

### Modified Files
- `projects/synth_tabular_classification/metrics/metrics.json` - Now tracked by DVC instead of Git

---

## Lessons Learned

1. **DVC 3.x Syntax**: Use relative paths from project root
2. **Metrics Not Cached**: Setting `cache: false` keeps metrics visible in Git
3. **Pipeline DAG**: `dvc dag` shows the dependency graph visually
4. **Reproducibility**: Changing any dependency triggers re-execution of downstream stages

---

## Next Sprint

**Sprint 2**: GCP & Terraform Infrastructure
- [HUMAN] 2.0.1-2.0.9: GCP account and Terraform setup
- [AI] 2.1-2.6: Terraform modules and configuration
