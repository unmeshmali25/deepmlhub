# MLOps Plan: Deep Learning Post-Training

## Goal

Create a reproducible post-training pipeline for the Qwen LoRA model. When the
training data changes, the pipeline should detect the change, run training on a
RunPod GPU, evaluate the result, and record the experiment.

The design should follow the general structure of
`projects/voice_offer_recommendation`, while replacing dbt and Feast with
Python-based JSONL preparation.

## Learning and Building Workflow

Build incrementally in local Jupyter notebooks, then move stable code into
`src/` rather than treating the notebook as the application:

1. Load and inspect the configuration.
2. Implement and test data preparation on a few examples.
3. Import `messages.py` and inspect the resulting datasets.
4. Run a tiny training example with one known model and PEFT method.
5. Move model and PEFT selection into configuration and factories.
6. Import the stable functions from `src/` back into the notebook and add
   focused tests.

Each step should be understood and verified before connecting the next one.

## Simple Architecture

```text
New or updated data.jsonl
          |
          v
      DVC pipeline
   prepare -> train -> evaluate
                |
                v
          RunPod GPU pod
                |
                v
             MLflow
```

## What Each Tool Does

| Tool | Responsibility |
| --- | --- |
| Git | Stores Python code, configuration, `dvc.yaml`, and `dvc.lock` |
| DVC | Versions large datasets and model artifacts, and detects pipeline changes |
| S3 or GCS | Stores DVC's large files and model outputs |
| RunPod | Provides the GPU environment for LoRA training |
| MLflow | Records parameters, losses, evaluation metrics, and artifacts |
| Docker | Packages the reproducible training environment |
| Docker Compose | Optional; useful for local MLflow/MinIO or future serving services |
| Feast | Not needed; this project has text examples, not online feature vectors |

## Proposed Repository Layout

```text
deeplearning_posttraining/
├── configs/
│   └── config.yaml
├── data/
│   ├── 01_cold_start_cot_sft/
│   │   ├── data.jsonl
│   │   └── eval.jsonl
│   └── processed/
├── evals/
├── metrics/
├── models/
├── scripts/
│   └── runpod_train.sh
├── src/llm_training/
│   ├── data/
│   │   └── prepare.py
│   ├── training/
│   │   └── sft_cot_lora.py
│   └── evaluation.py
├── Dockerfile
├── dvc.yaml
└── dvc.lock
```

## Pipeline Stages

### 1. Prepare Data

`messages.py` currently handles the in-memory preparation used by training: it loads JSONL, converts records to chat messages, and creates the split. A future `prepare.py` would be an optional DVC stage for validation, deduplication, and writing `data/processed/` outputs; it is not required by the current artifact.

Input: the source JSONL dataset.

Responsibilities:

- Validate every JSONL record against the expected schema.
- Remove malformed or duplicate records according to an explicit policy.
- Create deterministic training and evaluation files.
- Report row counts and validation failures.

Output examples:

- `data/processed/train.jsonl`
- `data/processed/eval.jsonl`
- `metrics/data_quality.json`

### 2. Train

Input: processed training JSONL and the training configuration.

Responsibilities:

- Load the Qwen base model.
- Run LoRA SFT using the existing training module.
- Log hyperparameters and training metrics to MLflow.
- Save the adapter and tokenizer as a versioned model artifact.

Output example:

- `models/adapter_qwen_cot/`

### 3. Evaluate

Input: the evaluation JSONL and the trained adapter.

Responsibilities:

- Evaluate the base model and fine-tuned model using the agreed rubric.
- Write machine-readable metrics.
- Generate comparison charts or other evaluation artifacts.
- Log metrics and artifacts to MLflow.

Output examples:

- `metrics/metrics.json`
- `evals/accuracy_chart.png`
- `evals/finetuned_scored.jsonl`

## DVC Design

`dvc.yaml` defines the pipeline commands, dependencies, and outputs. The
dataset should be an output of the data/preparation stage, or be separately
tracked with `dvc add` if it is manually maintained.

Example shape:

```yaml
stages:
  prepare_data:
    cmd: python -m llm_training.data.prepare
    deps:
      - data/01_cold_start_cot_sft/data.jsonl
      - src/llm_training/data/prepare.py
    outs:
      - data/processed/train.jsonl
      - data/processed/eval.jsonl

  train:
    cmd: python -m llm_training.training.sft_cot_lora
    deps:
      - data/processed/train.jsonl
      - configs/config.yaml
      - src/llm_training/training/sft_cot_lora.py
    outs:
      - models/adapter_qwen_cot

  evaluate:
    cmd: python -m llm_training.evaluation
    deps:
      - data/processed/eval.jsonl
      - models/adapter_qwen_cot
      - src/llm_training/evaluation.py
    outs:
      - metrics/metrics.json
      - evals/accuracy_chart.png
```

When `data.jsonl` changes, `dvc repro` notices the changed hash and reruns the
affected stages. If nothing changed, it skips the expensive training stage.

## RunPod Workflow

The RunPod pod should use a Docker image containing CUDA, PyTorch,
Transformers, PEFT, DVC, MLflow, and the project package.

On a fresh pod:

```bash
git clone <repository>
cd courses/deeplearning_posttraining
dvc pull
dvc repro
dvc push
```

- `git pull` gets code and the current pipeline snapshot.
- `dvc pull` downloads the exact data and prior artifacts referenced by the
  checked-out `dvc.lock`.
- `dvc repro` runs only stages whose inputs changed.
- `dvc push` uploads new DVC outputs to the configured object store.
- The updated `dvc.lock` must be committed to Git so the run is reproducible.

## Trigger Design

The first version can be manually started with the RunPod script. The target
automated flow is:

```text
New data population
        |
        v
Scheduled job or data-generation workflow
        |
        v
Commit data/DVC metadata and push to Git
        |
        v
GitHub Actions starts RunPod job
        |
        v
RunPod: git pull -> dvc pull -> dvc repro -> dvc push
```

Important: a data change alone does not magically start RunPod. A scheduler,
webhook, GitHub Actions workflow, or another orchestrator must invoke the
RunPod job. DVC decides whether training is necessary after the job starts.

## Configuration

Create one YAML configuration file for:

- Base model name.
- Data and artifact paths.
- LoRA parameters.
- Training parameters and random seed.
- Evaluation settings.
- MLflow experiment and tracking URI.

**Under consideration:** make the PEFT method configurable as well, rather than
hard-coding LoRA. A method field could select LoRA, AdaLoRA, IA3, or another
supported algorithm, with method-specific parameters validated before training.

Secrets must not be committed. Supply them to the RunPod environment through
RunPod secrets or environment variables, including:

- MLflow credentials.
- DVC remote credentials.
- Hugging Face token, if required.
- GitHub deploy credentials, if the pod pushes commits automatically.

## Docker Compose Decision

Docker Compose is not required for the RunPod training job. RunPod needs one
GPU-enabled training container, so a Dockerfile is sufficient.

Compose is optional for local development if we want to run services such as:

- A local MLflow tracking server.
- MinIO as a local S3-compatible DVC remote.
- A future vLLM server and API gateway.

## Implementation Phases

### Phase 1: Make the current training script pipeline-ready

- Add centralized configuration.
- Make input and output paths explicit.
- Ensure training exits non-zero on failure.
- Ensure outputs are written to stable artifact directories.

### Phase 2: Add data preparation and evaluation stages

- Extract validation and deterministic splitting from notebooks.
- Add repeatable evaluation code.
- Add tests for schema validation, splitting, and metric output.

### Phase 3: Add DVC locally

- Define `dvc.yaml`.
- Run `dvc repro` with a small test configuration.
- Confirm that unchanged inputs skip training.
- Confirm that a changed JSONL line reruns the required stages.

### Phase 4: Add remote artifact storage

- Configure an S3 or GCS DVC remote.
- Store credentials outside Git.
- Verify `dvc push` and `dvc pull` from a clean environment.

### Phase 5: Package and run on RunPod

- Build and test the GPU Docker image.
- Run the complete pipeline on a RunPod pod.
- Log a complete MLflow run.
- Confirm the adapter can be downloaded and loaded independently.

### Phase 6: Automate retraining

- Add a scheduled or event-driven GitHub Actions workflow.
- Start a RunPod job only after a new data/DVC commit.
- Push artifacts and publish the resulting `dvc.lock`.
- Add failure notifications and a concurrency guard so two training jobs do
  not run for the same data version.

## Definition of Done

- A clean checkout can restore data and model artifacts with `dvc pull`.
- `dvc repro` runs prepare, train, and evaluate in order.
- A data change causes a new training run; an unchanged dataset does not.
- Training and evaluation metrics appear in MLflow.
- The adapter and evaluation outputs are stored in the DVC remote.
- Git commit plus `dvc.lock` identifies the exact code, configuration, and
  dataset used for a run.
- The same pipeline runs successfully on a fresh RunPod GPU pod.
