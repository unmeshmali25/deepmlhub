# Active Sprint: GCP & Terraform Infrastructure

**Sprint**: sprint_02_terraform_infra  
**Last Updated**: 2026-02-04  
**Status**: 🔄 In Progress

---

## Sprint Goal

Create reusable Terraform infrastructure modules for GCP resources (GCS, MLflow on Cloud Run, Artifact Registry) to enable reproducible cloud deployments.

---

## What's Happening Now

### 🔄 In Progress
| Task | Assignee | Started | Status |
|------|----------|---------|--------|
| [AI] 2.1: Create Terraform Directory Structure | AI | 2026-02-04 | 🔄 In Progress |

### ⬜ Ready to Start
| Task | Assignee | Priority | Dependencies |
|------|----------|----------|--------------|
| [AI] 2.2: Create GCS Module | AI | High | AI 2.1 |
| [AI] 2.3: Create Artifact Registry Module | AI | Medium | AI 2.1 |
| [AI] 2.4: Create MLflow Cloud Run Module | AI | High | AI 2.1 |
| [AI] 2.6: Create Dev Environment Config | AI | High | AI 2.1-2.4 |

### ⬜ Optional/Low Priority
| Task | Assignee | Priority | Dependencies |
|------|----------|----------|--------------|
| [AI] 2.5: Create GKE Module | AI | Low | AI 2.1 |

### ✅ Completed Prerequisites
All [HUMAN] prerequisites are complete:
- GCP account with $300 credit
- Google Cloud CLI installed and authenticated
- GCP project created and configured
- Billing linked and alerts set ($20/month)
- 8 required APIs enabled
- Terraform CLI installed
- GCS state bucket created with versioning
- Terraform service account with IAM roles and JSON key

---

## Sprint Metrics

**Tasks**: 0/6 Complete (0%)  
**Blockers**: None  
**Next Milestone**: First Terraform module created

---

## Verification

When complete, run:
```bash
cd infrastructure/terraform/environments/dev
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your values

cp backend.tf.example backend.tf
# Edit backend.tf with your state bucket

terraform init
terraform plan
terraform apply
terraform output
```

---

## Quick Links

- [Master Backlog](backlog.md)
- [Sprint 02 Tasks](sprints/sprint_02_terraform_infra/tasks.md)
- [Architecture Plan](plans/mlops_plan_uno.md)
- [AGENTS.md](../AGENTS.md)
