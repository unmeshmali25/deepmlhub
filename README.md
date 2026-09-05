# deepmlhub

Personal MLOps monorepo: one flagship production-style project, focused
learning projects, and a real shared infra stack (Supabase, dbt, Feast,
DVC, MLflow, Terraform, GitHub Actions).

## Repo map

```
projects/        # Code. Each project is self-contained:
                 #   src/ tests/ configs/ data/ models/ dvc.yaml
resources/       # Gitignored reference material: books, PDFs, notes.
infrastructure/  # Shared infra: Terraform (GKE, GCS, Cloud SQL, MLflow, VPC)
                 # + Dockerfiles for shared services.
.github/         # CI/CD — workflows trigger per-project via path filters.
```

## Projects

| Project | Purpose | Stack | Status |
|---|---|---|---|
| `voice_offer_recommendation` | Offer recommender for simulated retail agents (flagship) | Supabase, dbt, Feast, DVC, MLflow, GCP | E2E pipeline green (17/17 pytest, 70/70 dbt tests) |
| `llm-retard-lab` | LLM post-training: SFT/DPO/RLHF experiments + evals | PyTorch, PEFT/LoRA, Qwen | Cold-start CoT adapter trained + evaluated || `synth_tabular_classification` | Baseline ML pipeline pattern | sklearn, DVC, FastAPI | Complete |
| `pytorch_00..03` | PyTorch fundamentals curriculum | PyTorch | Complete |
| `microprojects` | Small focused experiments (async, SQLAlchemy, inference) | varies | Ongoing |

## Artifact policy — what's tracked where

- **git**: code, configs, notebooks, docs, DVC pointer files (`.dvc`)
- **DVC**: model weights and other large artifacts (e.g. the trained
  LoRA adapter at `projects/llm-retard-lab/models/adapter_qwen_cot_full`)
- **gitignored `resources/`**: consumed reference material (epub, lecture PDFs)
- **never tracked**: `.env` (see `.env.example` where relevant), caches,
  old training runs, `mlflow.db`

Note: older git history contains large model binaries from before this
policy. Future commits stay lean.

## Quickstart

```bash
# Flagship ML pipeline (dbt -> Feast -> DVC -> MLflow)
cd projects/voice_offer_recommendation
source ../../.venv/bin/activate && source .env
cd dbt && dbt run && dbt test && cd ..
dvc repro

# PyTorch curriculum
cd projects/pytorch_00_tensor_gpu_basics
python src/tensor_basics.py
```

## CI

Each project owns its workflow in `.github/workflows/`, triggered only
when files under that project change.
