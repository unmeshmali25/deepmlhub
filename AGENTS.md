# AGENTS.md - Guidelines for Agentic Coding Agents

This repository contains PyTorch-based machine learning practice projects and a synthetic tabular classification project using scikit-learn.

## Current Project Status (Updated 2026-04-23)

### Voice Offer Recommendation — Sprint 10: dbt Feature Pipeline + E2E Pipeline ✅ COMPLETE

**Status**: Sprint 10 and Sprint 9 verification fully complete. End-to-end pipeline passes.

**What's working**:
- dbt ETL: 6 staging views → 4 intermediate tables → 3 feature tables in `dbt_vor` schema
- Feature tables: `fct_agent_features` (371 rows), `fct_product_features` (72 rows), `fct_agent_product_interactions` (9,106 rows)
- DVC pipeline: `transform_features` → `fetch_features` → `train` → `evaluate` — ALL 4 STAGES PASS
- MLflow: Training + evaluation metrics logged to remote server
- Training data: 9,106 rows × 82 features in parquet
- Model: SimpleRuleRecommender trained and saved
- Tests: 17/17 pytest, 70/70 dbt tests

**What's next**:
- Sprint 11: Replace `SimpleRuleRecommender` with collaborative filtering

**Key gotchas (learned during verification)**:
- `.env` needs `export` prefix for dbt `env_var()` to work
- `dbt deps` must run before first `dbt run`
- `agent_state_snapshots.agent_id` is UUID FK to `agents.id` — must JOIN to resolve to `agents.agent_id` (varchar)
- Cast all UUID columns to varchar in dbt staging (not just `cast()`, need actual JOIN resolution)
- `+schema` config creates sub-schemas — remove to keep everything in single `dbt_vor` schema
- PostgreSQL UUID columns come through as Python UUID objects — pyarrow can't serialize to Parquet
- MLflow rejects `@` in metric names — use `precision_at_5` not `precision@5`
- `protobuf>=5` breaks MLflow 2.10 — pin to `<5`
- Actual data values: `coupon_interactions.action` = `added_to_cart/redeemed`, `user_coupons.status` = `active/expired/used`
- pgBouncer hostname: `aws-1-us-east-1` (not `aws-0`)

**Quick commands**:
```bash
cd projects/voice_offer_recommendation
source ../../.venv/bin/activate && source .env
cd dbt && dbt run && dbt test     # ETL pipeline
cd .. && dvc repro                # Full ML pipeline
```

## Build/Lint/Test Commands

### Running Tests

```bash
# Run all tests in a project
cd projects/synth_tabular_classification
pytest

# Run a single test file
pytest tests/test_model.py

# Run a single test class
pytest tests/test_model.py::TestModel

# Run a single test method
pytest tests/test_model.py::TestModel::test_random_forest_trains

# Run tests with coverage
pytest --cov=src --cov-report=term-missing

# Run tests in verbose mode
pytest -v
```

### Linting and Type Checking

```bash
# Run ruff linter (linting + formatting)
ruff check .
ruff check --fix .  # Auto-fix issues

# Run ruff format
ruff format .

# Run mypy type checker
mypy src/

# Check all quality gates
ruff check . && mypy src/ && pytest
```

### Running Python Scripts

```bash
# Navigate to a project and run a script
cd projects/pytorch_00_tensor_gpu_basics
python src/tensor_basics.py

# For PyTorch projects with GPU support
python src/tensor_basics.py  # Auto-detects CUDA
```

## Code Style Guidelines

### Python Style

- **Formatter**: Use `ruff` for both linting and formatting
- **Line length**: 88 characters (Black-compatible)
- **Quote style**: Double quotes for strings
- **Import sorting**: Use standard library, third-party, local imports

### Import Conventions

```python
# Standard library imports first
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional

# Third-party imports second
import numpy as np
import pandas as pd
import torch
import yaml
from fastapi import FastAPI
from pydantic import BaseModel

# Local imports last
from src.data.generate import generate_data
from src.model.train import train_model
```

### Naming Conventions

- **Functions**: `snake_case` - `train_model()`, `load_config()`
- **Classes**: `PascalCase` - `TestModel`, `PredictionRequest`
- **Constants**: `UPPER_SNAKE_CASE` - `RANDOM_SEED`, `BATCH_SIZE`
- **Private methods**: `_leading_underscore` - `_internal_helper()`
- **Test classes**: `Test` prefix - `TestModel`, `TestInferenceEndpoints`
- **Test methods**: Descriptive `snake_case` - `test_random_forest_trains()`

### Type Hints

- Use type hints for all function parameters and return values
- Use `Optional[T]` for nullable values
- Use `List[T]`, `Dict[K, V]` from typing module
- Use `-> None` for functions that don't return anything

```python
def load_config() -> dict:
    """Load configuration from config.yaml."""
    ...

def train_model(data_path: Path, epochs: int = 10) -> tuple:
    """Train model and return model with metrics."""
    ...
```

### Docstrings

- Use triple double quotes `"""`
- First line: Brief description (imperative mood)
- Include docstrings for all public modules, functions, classes, methods

```python
def measure_device_transfer_overhead():
    """
    Exercise 2: Measure CPU <-> GPU Transfer Overhead

    Questions to explore:
    - How expensive is CPU-GPU data transfer?
    - Does transfer time scale linearly with tensor size?
    """
```

### Error Handling

- Use specific exceptions (not bare `except:`)
- Provide informative error messages
- Use `try/except` blocks for expected failure cases

```python
try:
    result = cpu_tensor + gpu_tensor
except RuntimeError as e:
    print(f"Error: {e}")
    print("Lesson: All tensors in an operation must be on the same device!")
```

### Testing Conventions

- Use `pytest` framework
- Organize tests in `tests/` directory with `__init__.py`
- Group related tests in classes: `class TestModel:`
- Use descriptive test names that explain what is being tested
- Use fixtures for common setup (when needed)

```python
class TestModel:
    def test_random_forest_trains(self):
        """Test that RandomForest can train on dummy data."""
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)
        assert len(predictions) == 100
```

## Project Structure

```
projects/
├── pytorch_00_tensor_gpu_basics/    # PyTorch fundamentals
├── pytorch_01_neural_networks/      # Custom modules
├── pytorch_02_training_optimization/# Training loops
├── pytorch_03_computer_vision/      # CNNs & CIFAR-10
└── synth_tabular_classification/    # ML pipeline example
    ├── src/
    │   ├── data/          # Data generation and preprocessing
    │   ├── model/         # Training and evaluation
    │   └── inference/     # FastAPI server and prediction
    ├── tests/             # Unit tests
    ├── configs/           # YAML configurations
    ├── data/              # Training data
    ├── models/            # Saved model artifacts
    └── metrics/           # Evaluation metrics
```

## Dependencies

- PyTorch projects use `torch>=2.0.0` with CUDA support
- ML projects use `scikit-learn>=1.5.0`
- API projects use `fastapi>=0.109.0`
- Testing uses `pytest>=7.4.0`
- Linting uses `ruff>=0.1.0` and `mypy>=1.7.0`

## Environment Setup

```bash
# Use the existing virtual environment
source .venv/bin/activate

# Or install dependencies per project
cd projects/<project_name>
pip install -r requirements.txt
```

## Key Conventions

1. **GPU Handling**: Always check `torch.cuda.is_available()` before GPU operations
2. **Device Management**: Use `.to(device)` pattern for flexible device placement
3. **MLflow Tracking**: Log parameters, metrics, and models in training scripts
4. **DVC Integration**: Track data and model artifacts with DVC
5. **FastAPI**: Use Pydantic models for request/response validation
6. **Random Seeds**: Set `random_state` for reproducibility in ML models

## Working with Plans and TODOs

This repository uses `.agents/` directory for tracking complex multi-phase work with a structured approach.

### Quick Start

1. **Start with CURRENT_FOCUS.md** - See what's happening today
2. **Check active tasks** - View `mlops_todos_current.md` for details
3. **Archive completed work** - Move finished phases to `archive/` subdirectories
4. **Update status immediately** - After completing any task

### File Structure

```
.agents/
├── CURRENT_FOCUS.md              # What's happening NOW (start here)
├── plans/
│   ├── mlops_plan_uno.md        # Architecture/design documents
│   └── archive/                 # Completed plans
├── todos/
│   ├── mlops_todos_current.md   # Active tasks only
│   └── archive/                 # Completed phase archives
└── README.md                    # This documentation
```

### Active Plans

- **Architecture**: `.agents/plans/mlops_plan_uno.md`
- **Current Tasks**: `.agents/todos/mlops_todos_current.md`
- **Today's Focus**: `.agents/CURRENT_FOCUS.md`

### Marking Tasks Complete

**After completing ANY task:**

1. **Update the task status** in `mlops_todos_current.md`:
   - Change emoji to `✅`
   - Add completion timestamp: `**Completed**: 2026-01-30`
   - List what was delivered

2. **Update CURRENT_FOCUS.md**:
   - Mark task complete in progress table
   - Update phase completion percentage
   - Move to next task or note blockers

3. **Archive completed phases** (when entire phase done):
   - Copy phase to `todos/archive/phase_N_name.md`
   - Add lessons learned section
   - Remove from active todos

**Example:**
```markdown
#### AI 1.1: Create DVC Pipeline File ✅

**Status**: ✅ Complete (2026-01-30)
**Time Taken**: 45 minutes

**Delivered**:
- Created `projects/synth_tabular_classification/dvc.yaml`
- 4 stages: generate, preprocess, train, evaluate
- Verified with `dvc dag`

**Files Changed**:
- `projects/synth_tabular_classification/dvc.yaml` (new)
```

### Finding Human Action Items

Human tasks are tracked in multiple places:

1. **CURRENT_FOCUS.md** → "Human Action Items" section
   - Table of upcoming tasks with time estimates
   - Links to full details in todos file
   - Shows what's blocking AI work

2. **mlops_todos_current.md** → Each phase has "Human Prerequisites"
   - Step-by-step instructions
   - Verification commands
   - Definition of Done checkboxes

3. **Blocked tasks** reference human prerequisites:
   ```markdown
   🚫 Blocked By: [Human 1.1-1.6](#upcoming-phase-2-prerequisites-)
   ```

### Modifying Plans During Implementation

**When reality diverges:**

**Small tweaks** (single task):
- Edit task description directly
- Add note: `**Amendment**: Changed X to Y because Z`
- Update dependencies if needed

**Significant changes** (new phases, architecture):
- **Option A**: Create new plan file (e.g., `mlops_plan_v2.md`)
- **Option B**: Add "Amendments" section to existing plan

**Required change documentation**:
```markdown
### Amendment 1: Changed from X to Y
**Date**: 2026-01-30
**Reason**: [Why change was needed]
**Impact**: [What this affects]
**Updated by**: AI/Human
```

### Plan Evolution Strategy

```
Phase N (Planning)
  ↓
Phase N+1 (Implementation starts)
  ↓
[Discoveries made, assumptions wrong]
  ↓
Update CURRENT_FOCUS → Continue
  ↓
Archive Phase N+1 → Phase N+2
```

**Guidelines**:
- Document wrong assumptions
- Keep original estimates for post-mortem
- Mark deprecated sections with strikethrough
- Notify human of significant restructuring

### Task Dependencies and Blocking

**Before starting any task:**

1. Check CURRENT_FOCUS.md for blockers
2. Verify human prerequisites are ✅ (not ⬜ or 🔄)
3. If blocked by human work:
   - Change status to `🚫`
   - Add note: "Blocked: Waiting for Human X.Y"
   - Ask human to complete prerequisite
   - Move to next unblocked task

**Example blocker documentation**:
```markdown
#### AI 2.1: Create Terraform Directory Structure 🚫

**Status**: 🚫 Blocked (2026-01-30)
**Blocked By**: Human 1.1-1.6, 2.1-2.3 (GCP setup)
**ETA**: After ~2.5 hours of human work

**Human needs to**:
1. Create GCP account
2. Install gcloud CLI
3. Create project and enable APIs
...
```

### Best Practices

1. **Commit plan changes**: `git commit -m "docs: mark AI 1.1 complete, start AI 1.2"`
2. **Use timestamps**: Track velocity and identify stale work
3. **Archive completed phases**: Reduce noise, preserve history
4. **Weekly review**: Scan for tasks stuck in `🔄` > 7 days
5. **Link to code**: Reference specific files when marking complete
6. **Update CURRENT_FOCUS first**: Always reflect current reality
