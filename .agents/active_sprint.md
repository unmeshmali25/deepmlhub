# Active Sprint: DVC Remote Setup

**Sprint**: sprint_04_dvc_remote  
**Last Updated**: 2026-02-05  
**Status**: 🔄 Ready to Start

---

## Sprint Goal

Configure DVC to use GCS as remote storage for versioning data and model artifacts. This enables collaboration and ensures data lineage tracking across different environments.

---

## What's Happening Now

### ⬜ Prerequisites (Pending Human)
| Task | Assignee | Status | Dependencies |
|------|----------|--------|--------------|
| [HUMAN] 4.1: Create GCS Bucket for DVC | Human | ⬜ Not Started | GCP Infrastructure |
| [HUMAN] 4.2: Configure DVC Remote | Human | ⬜ Not Started | HUMAN 4.1 |

### ⬜ Ready to Start (Blocked by Prerequisites)
| Task | Assignee | Priority | Dependencies |
|------|----------|----------|--------------|
| [AI] 4.1: Test DVC Remote Connection | AI | High | All HUMAN 4.x |
| [AI] 4.2: Document DVC Workflow | AI | Medium | All HUMAN 4.x |

---

## Sprint Metrics

**Tasks**: 0/4 Complete (0%)  
**Blockers**: [HUMAN] Prerequisites not started  
**ETA**: ~30 minutes after human prerequisites complete

---

## Prerequisites Detail

### [HUMAN] 4.1: Create GCS Bucket for DVC

**Tasks**:
1. Create bucket: `gs://deepmlhub-voiceoffers-dvc`
2. Enable uniform bucket-level access
3. Grant service account access

**Command**:
```bash
gcloud storage buckets create gs://deepmlhub-voiceoffers-dvc \
  --project=deepmlhub-voiceoffers \
  --location=us-central1 \
  --uniform-bucket-level-access
```

**Verification**: Bucket appears in Cloud Storage console

---

### [HUMAN] 4.2: Configure DVC Remote

**Tasks**:
1. Add remote to DVC config
2. Set as default remote
3. Test connection

**Command**:
```bash
cd projects/synth_tabular_classification
dvc remote add -d gcs gs://deepmlhub-voiceoffers-dvc
dvc remote modify gcs credentialpath ~/.config/gcloud/application_default_credentials.json
dvc push  # Test connection
```

**Verification**: `dvc push` succeeds without errors

---

## Next Steps

Once prerequisites are complete:
1. AI will test DVC remote connection
2. Document DVC workflow for the project
3. Push existing data to remote

---

## Quick Links

- [Sprint Tasks](sprints/sprint_04_dvc_remote/tasks.md)
- [Master Backlog](backlog.md)
- [Previous Sprint: GitHub Setup](sprints/sprint_03_github_setup/tasks.md)
- [Architecture Plan](plans/mlops_plan_uno.md)
