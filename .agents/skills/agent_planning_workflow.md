# Agent Planning Workflow - Complete Guide

**Purpose**: Exhaustive template for creating and managing project plans and todos  
**Scope**: Any multi-step project with dependencies, phases, and human/AI collaboration  
**Audience**: AI agents of all capabilities levels  

---

## Table of Contents

1. [When to Use This System](#when-to-use-this-system)
2. [File Structure](#file-structure)
3. [Phase 1: Initial Setup](#phase-1-initial-setup)
4. [Phase 2: Create Master Plan](#phase-2-create-master-plan)
5. [Phase 3: Create Initial Todos](#phase-3-create-initial-todos)
6. [Phase 4: Create CURRENT_FOCUS](#phase-4-create-current_focus)
7. [Daily Workflow](#daily-workflow)
8. [Phase Completion Protocol](#phase-completion-protocol)
9. [Plan Evolution Guidelines](#plan-evolution-guidelines)
10. [Critical Rules](#critical-rules)
11. [Error Handling](#error-handling)
12. [Templates Quick Reference](#templates-quick-reference)

---

## When to Use This System

Use this workflow for any project with:
- More than 3 distinct tasks
- Dependencies between tasks
- Both human and AI work
- Multiple phases or milestones
- Complex infrastructure setup
- Duration > 1 day

**Don't use for**: Simple single-file changes, bug fixes, or documentation updates

---

## File Structure

```
.agents/
├── CURRENT_FOCUS.md              # START HERE - Today's priorities
├── README.md                     # Documentation for this directory
├── plans/
│   ├── master_plan.md           # Architecture/design document
│   └── archive/                 # Completed/archived plans
├── todos/
│   ├── current_tasks.md         # Active tasks (current phase only)
│   └── archive/                 # Completed phase archives
│       └── phase_0_name.md
└── skills/                      # Reusable skill definitions
```

### File Purposes

**CURRENT_FOCUS.md**: Daily starting point showing:
- What's in progress 🔄
- What's blocked 🚫
- What's ready to start ⬜
- Today's specific goals
- Phase completion percentages
- Human action items (upcoming and current)

**plans/master_plan.md**: High-level architecture:
- Technology decisions
- Component diagrams
- Cost estimates
- Security considerations
- Implementation approach

**todos/current_tasks.md**: Detailed execution plan:
- Current phase tasks only
- Definition of Done for each task
- Verification commands
- Dependencies and blockers
- Estimated time and priority

**todos/archive/**: Completed phases:
- Full phase history
- Lessons learned
- What worked/didn't work
- Reference for future similar work

---

## Phase 1: Initial Setup

### Step 1.1: Create .agents directory structure

**Command**:
```bash
mkdir -p .agents/plans/archive
mkdir -p .agents/todos/archive
mkdir -p .agents/skills
touch .agents/README.md
touch .agents/CURRENT_FOCUS.md
```

**Verification**:
```bash
ls -la .agents/
# Should show: plans, todos, skills, README.md, CURRENT_FOCUS.md
```

**Success criteria**: All directories and files created

**Error handling**: 
- If directories already exist → Continue (that's OK)
- If permission denied → Ask human for write permissions
- If `mkdir` fails → Check disk space

**Time estimate**: 1 minute

---

## Phase 2: Create Master Plan

### Step 2.1: Create master plan document

**File**: `.agents/plans/master_plan.md`

**Required sections** (in order):

#### 1. Title and Overview
```markdown
# [Project Name] Master Plan

## Overview
[One paragraph describing:
- What we're building
- Why we're building it
- Success criteria
- Timeline estimate
]

**Scope**: [What's in scope]
**Out of Scope**: [What's explicitly not included]
```

**Example**:
```markdown
# MLOps Infrastructure Plan

## Overview
Build a complete MLOps infrastructure on GCP for a solo developer managing 
10 ML projects. Optimized for minimal cost (~$5-15/month baseline) while 
providing production-ready capabilities including experiment tracking, 
data versioning, and scalable training.

**Scope**: GCP infrastructure, Terraform IaC, MLflow tracking, DVC, GKE cluster
**Out of Scope**: Multi-region deployment, enterprise security, 24/7 monitoring
```

#### 2. Architecture
```markdown
## Architecture

### High-Level Diagram
```
[ASCII art diagram showing components and connections]
```

### Component Overview

| Component | Technology | Purpose |
|-----------|------------|---------|
| [Name] | [Tech] | [What it does] |
```

**Example**:
```markdown
## Architecture

```
┌─────────────────┐     ┌─────────────────┐
│   Cloud Run     │────▶│   GCS Bucket    │
│  MLflow Server  │     │  DVC + Artifacts│
└─────────────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐
│   GKE Cluster   │
│  (Training)     │
└─────────────────┘
```

| Component | Technology | Purpose |
|-----------|------------|---------|
| Experiment Tracking | MLflow on Cloud Run | Track experiments, models, metrics |
| Data Versioning | DVC with GCS | Version control for datasets |
| Training Platform | GKE with Spot VMs | Scalable model training |
```

#### 3. Technology Decisions
```markdown
## Technology Decisions

| Component | Choice | Rationale | Alternatives Considered |
|-----------|--------|-----------|------------------------|
| [Component] | [Technology] | [Why this choice] | [What else was considered] |
```

**Example**:
```markdown
| Component | Choice | Rationale | Alternatives |
|-----------|--------|-----------|--------------|
| MLflow Hosting | Cloud Run | Serverless, scales to zero, cheap | Cloud SQL (too expensive) |
| Container Registry | Artifact Registry | Native GCP integration | Docker Hub (external) |
| Orchestration | GKE Standard | Free control plane, Spot VMs | GKE Autopilot (more $) |
```

#### 4. Directory Structure
```markdown
## Directory Structure

```
project-root/
├── folder/
│   ├── subfolder/
│   │   └── file.ext
│   └── file.ext
└── file.ext
```
```

#### 5. Component Details

For each major component, create a subsection:

```markdown
### 1. [Component Name]

**Purpose**: [One sentence description]

**Architecture**:
- [Bullet points describing how it works]
- [Key features]
- [Integration points]

**Configuration**:
```yaml
# Example config or code
key: value
```

**Files**:
- `path/to/main.tf` - [What this file does]
- `path/to/config.yaml` - [What this file does]

**Cost**: [Estimated monthly cost]

**Security Considerations**:
- [Security point 1]
- [Security point 2]
```

#### 6. Cost Estimation (if applicable)
```markdown
## Cost Estimation

| Service | Monthly Cost | Notes |
|---------|--------------|-------|
| [Service] | $X | [Assumptions] |
| **Total** | **$X** | |

**Cost optimization strategies**:
- [Strategy 1]
- [Strategy 2]
```

#### 7. Security Considerations
```markdown
## Security Considerations

1. **[Risk]**: [Description]
   - **Mitigation**: [How to address]
   
2. **[Risk]**: [Description]
   - **Mitigation**: [How to address]
```

### Step 2.2: Verification

**Checklist**:
- [ ] All 7 required sections present
- [ ] Architecture diagram is readable
- [ ] Technology decisions have rationale
- [ ] Directory structure matches project layout
- [ ] Component details have examples/configs

**Success criteria**: Plan document is complete and could be used by another engineer to understand the project

**Error handling**:
- Missing sections → Add them
- Unclear diagrams → Redraw with clearer ASCII art
- Missing rationale → Research and document why choices were made

**Time estimate**: 30-60 minutes depending on project complexity

---

## Phase 3: Create Initial Todos

### Step 3.1: Create current_tasks.md

**File**: `.agents/todos/current_tasks.md`

**Required format**:

```markdown
# [Project] - Active Tasks

**Current Phase**: Phase X - [Name]
**Last Updated**: YYYY-MM-DD
**Last Updated By**: AI/Human

---

## Status Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Complete |
| 🔄 | In Progress |
| ⬜ | Not Started |
| ⏭️ | Skipped |
| 🚫 | Blocked |

---

## Phase X: [Phase Name]

**Status**: 🔄 In Progress / ⬜ Not Started
**Started**: YYYY-MM-DD (if in progress)
**Target Completion**: YYYY-MM-DD
**Prerequisites**: [Previous phase name or "None"]

### Overview
[2-3 sentences describing what this phase accomplishes and why it matters]

### Success Criteria
- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Criterion 3

---

### Tasks

#### [AI/Human] X.Y: [Task Name] [STATUS]

**Status**: [STATUS emoji] [Status text] (YYYY-MM-DD)
**Estimated Time**: XX minutes
**Priority**: High / Medium / Low
**Blocked By**: [Task reference, e.g., "AI X.Y-1"] or "None"

**Definition of Done**:
- [ ] Specific criterion 1
- [ ] Specific criterion 2
- [ ] Specific criterion 3

**Steps**:
```bash
# Command 1 with explanation
# Command 2 with explanation
# Command 3 with explanation
```

**Verification**:
```bash
# Command to verify completion
# Expected output
```

**Files Created/Modified**:
- `path/to/file1` (new / modified)
- `path/to/file2` (new / modified)

**Notes**:
[Any special considerations, edge cases, or context]

**Amendments** (if any):
- **YYYY-MM-DD**: [Description of change made]

---

[Repeat task template for each task in phase]

---

### Phase X Verification

**Prerequisites**: All tasks in Phase X marked ✅

**Verification steps**:
```bash
# Step 1
# Step 2
# Step 3
```

**Success criteria**:
- [ ] All tasks complete
- [ ] Verification steps pass
- [ ] No errors or warnings
- [ ] Documentation updated

```

### Step 3.2: Task Template Guidelines

**Good task characteristics**:

✅ **GOOD**: "Create DVC Pipeline File"
- Clear what needs to be done
- Has specific criteria
- Includes verification

❌ **BAD**: "Set up DVC"
- Too vague
- No criteria for "done"
- No verification steps

**Definition of Done must be**:
- Specific (not "set up X" but "create file Y with Z contents")
- Measurable (can verify with a command)
- Achievable (within the estimated time)
- Relevant (contributes to phase goal)
- Time-bound (has estimated duration)

### Step 3.3: Verification

**Checklist**:
- [ ] At least one task has Definition of Done with 3-5 checkboxes
- [ ] Each task has estimated time
- [ ] Dependencies are documented
- [ ] Verification commands are provided
- [ ] Status is set to ⬜ for all tasks initially

**Success criteria**: Could hand this to another agent and they could execute without asking questions

**Error handling**:
- Vague tasks → Break down into smaller, specific tasks
- Missing verification → Add verification commands
- No dependencies → Analyze task relationships and document them

**Time estimate**: 20-40 minutes depending on number of tasks

---

## Phase 4: Create CURRENT_FOCUS

### Step 4.1: Create CURRENT_FOCUS.md

**File**: `.agents/CURRENT_FOCUS.md`

**Required sections**:

```markdown
# Current Focus

**Last Updated**: YYYY-MM-DD HH:MM
**Active Phase**: Phase X - [Name]
**Next Milestone**: [What's coming up after this phase]

---

## What's Happening Now

### In Progress 🔄

| Task | Phase | Assigned | Started |
|------|-------|----------|---------|
| [Task name with link] | Phase X | AI/Human | YYYY-MM-DD |

### Blocked 🚫

| Task | Blocked By | ETA |
|------|------------|-----|
| [Task name with link] | [Human X.Y or AI X.Y] | [When unblocked] |

### Ready to Start ⬜

| Task | Phase | Priority |
|------|-------|----------|
| [Task name with link] | Phase X | High/Medium/Low |

---

## Today's Goal

Complete these items today:
1. [ ] Task 1
2. [ ] Task 2
3. [ ] Task 3

---

## Phase Progress

| Phase | Status | Tasks Complete | Total Tasks | Progress |
|-------|--------|----------------|-------------|----------|
| Phase 0: [Name] | ✅ Complete | X/Y | 100% |
| Phase 1: [Name] | 🔄 In Progress | X/Y | Z% |
| Phase 2: [Name] | 🚫 Blocked | X/Y | Z% |
| Phase 3: [Name] | ⬜ Not Started | X/Y | 0% |

---

## Human Action Items

### Current: Phase X [COLOR]
[If human tasks are currently needed]

| Task | Phase | Est. Time | Priority | Status |
|------|-------|-----------|----------|--------|
| [Human X.Y: Task name with link] | Phase X | XX min | High/Medium/Low | ⬜ Not Started |

### Upcoming: Phase Y Prerequisites [COLOR]
[If human tasks coming up soon]

| Task | Phase | Est. Time | Priority | Details |
|------|-------|-----------|----------|---------|
| [Human Y.Y: Task name with link] | Phase Y | XX min | High/Medium/Low | [Brief description] |

**Total Phase Y human tasks**: N  
**Estimated total time**: ~X hours  
**Current Status**: [Status message]

---

## Notes

- [Note 1]
- [Note 2]

---

## Quick Links

- [Phase 0 Archive](../todos/archive/phase_0_name.md)
- [Current Tasks](../todos/current_tasks.md)
- [Master Plan](../plans/master_plan.md)
- [AGENTS.md](../../AGENTS.md)
```

### Step 4.2: Current Focus Best Practices

**Update frequency**:
- **Morning**: Read to understand priorities
- **After each task**: Update status and progress
- **End of day**: Update "Today's Goal" for tomorrow
- **Phase completion**: Update phase table

**Human Action Items section**:
- Always include links to full task details
- Show estimated time so human can plan
- Color code priority (🔴 Critical, 🟡 Recommended, 🟢 Optional)
- Indicate what's blocked until human acts

---

## Daily Workflow

### Morning Routine (5 minutes)

**Step 1**: Read CURRENT_FOCUS.md
```bash
cat .agents/CURRENT_FOCUS.md
```

**What to look for**:
- What's in progress 🔄 (finish this first)
- What's blocked 🚫 (can you unblock it?)
- What's ready to start ⬜ (pick highest priority)
- Human action items (remind human if needed)

**Step 2**: Check for blockers
- Are any AI tasks blocked by human work?
- Are prerequisites complete?
- If blocked, can you work on something else?

**Step 3**: Pick next task
- Highest priority unblocked task
- Update its status to 🔄
- Add start timestamp: `**Started**: YYYY-MM-DD HH:MM`

### During Work (Ongoing)

**When starting a task**:
1. Update status to 🔄
2. Add timestamp
3. Read full task details in todos/current_tasks.md
4. Execute steps

**When completing a task**:
1. Run verification commands
2. Verify all Definition of Done checkboxes
3. Update status to ✅
4. Add completion timestamp: `**Completed**: YYYY-MM-DD HH:MM`
5. Document what was delivered
6. Update CURRENT_FOCUS.md

**When blocked**:
1. Change status to 🚫
2. Add note: "Blocked: Waiting for [Human/AI X.Y]"
3. Document what you need
4. Ask human if human task is blocking
5. Move to next unblocked task

### End of Day (5 minutes)

**Step 1**: Commit all changes
```bash
git add .agents/
git commit -m "docs: [phase] [action] - [brief description]"

# Examples:
git commit -m "docs: Phase 1 - mark AI 1.1 complete, start AI 1.2"
git commit -m "docs: Phase 2 - add amendment for GCS module"
git commit -m "docs: archive Phase 0, start Phase 1"
```

**Step 2**: Update phase completion percentage
- Count completed tasks vs total
- Update CURRENT_FOCUS.md phase progress table

**Step 3**: Check for stale tasks
- Any task in 🔄 status >7 days?
- Add note: "⏰ Stale - in progress >7 days"
- Escalate to human

---

## Phase Completion Protocol

**When ALL tasks in a phase are ✅**:

### Step 1: Archive the Phase

**Copy phase to archive**:
```bash
cp .agents/todos/current_tasks.md .agents/todos/archive/phase_X_name.md
```

**Add archive header** at the top of archived file:

```markdown
# Phase X: [Name] [ARCHIVED]

**Status**: ✅ Complete  
**Completed**: YYYY-MM-DD  
**Archived**: YYYY-MM-DD  
**Total Duration**: X days/weeks  

---

## Summary

[2-3 sentences describing what this phase accomplished and its impact on the overall project]

## Key Deliverables

- [Deliverable 1]: [Brief description]
- [Deliverable 2]: [Brief description]
- [Deliverable 3]: [Brief description]

## Metrics

- Tasks completed: X/Y
- Estimated time: X hours
- Actual time: X hours
- Blockers encountered: N

## Lessons Learned

### What Worked Well
1. [Success 1 - why it worked]
2. [Success 2 - why it worked]
3. [Success 3 - why it worked]

### What Could Be Improved
1. [Improvement 1 - what went wrong and how to fix next time]
2. [Improvement 2 - what went wrong and how to fix next time]

### Key Insights
- [Insight 1 - something you learned]
- [Insight 2 - something you learned]

## Files Created

- `path/to/file1` - [Purpose]
- `path/to/file2` - [Purpose]

---

[Rest of original phase content follows...]
```

### Step 2: Update current_tasks.md

**Remove completed phase tasks**:
- Delete all Phase X task sections
- Keep only the next phase's tasks

**Update headers**:
```markdown
# [Project] - Active Tasks

**Current Phase**: Phase Y - [Next Phase Name]
**Last Updated**: YYYY-MM-DD
```

### Step 3: Update CURRENT_FOCUS.md

**Mark phase as ✅**:
```markdown
| Phase | Status | Tasks Complete | Total | Progress |
|-------|--------|----------------|-------|----------|
| Phase X: [Name] | ✅ Complete | Y/Y | 100% |
| Phase Y: [Name] | 🔄 In Progress | 0/Y | 0% |
```

**Update Active Phase**:
```markdown
**Active Phase**: Phase Y - [Next Phase Name]
```

**Set Today's Goal**:
```markdown
## Today's Goal

Start Phase Y:
1. [ ] Read Phase Y overview
2. [ ] Complete task Y.1
3. [ ] Update documentation
```

### Step 4: Commit Archive

```bash
git add .agents/
git commit -m "docs: archive Phase X, start Phase Y"
```

---

## Plan Evolution Guidelines

### Level 1: Small Changes (Single Task)

**When**: Minor adjustment to one task

**Process**:
1. Edit task directly in current_tasks.md
2. Add amendment note at bottom of task:
   ```markdown
   **Amendment**: Changed X to Y because Z
   **Date**: YYYY-MM-DD
   **Updated by**: AI
   ```
3. Update dependencies if affected
4. Commit: `git commit -m "docs: amend task X.Y - reason"`

**Examples**:
- Change estimated time from 30min to 45min
- Add new verification step
- Update file path

### Level 2: Medium Changes (Multiple Tasks)

**When**: Changes affect several tasks in a phase

**Process**:
1. Add "Amendments" section to current_tasks.md before the tasks
   ```markdown
   ## Amendments
   
   ### Amendment 1: [Brief description]
   **Date**: YYYY-MM-DD
   **Reason**: [Why change was needed]
   **Impact**: [What this affects]
   **Changes Made**:
   - [Change 1 description]
   - [Change 2 description]
   - [Change 3 description]
   
   **Updated by**: AI/Human
   ```
2. Update all affected tasks
3. Update CURRENT_FOCUS if priorities changed
4. Commit: `git commit -m "docs: amendments for Phase X - reason"`

**Examples**:
- Reordering tasks due to discovered dependency
- Adding 2-3 new tasks
- Removing obsolete tasks

### Level 3: Large Changes (Architecture, New Phases)

**When**: Fundamental changes to plan structure

**Option A: Create New Plan Version**
```bash
# Copy existing plan
cp .agents/plans/master_plan.md .agents/plans/master_plan_v2.md

# Update v2 with changes
echo "# [Project] Master Plan v2" > .agents/plans/master_plan_v2.md
# ... add updated content ...

# Archive old plan
mv .agents/plans/master_plan.md .agents/plans/archive/master_plan_v1.md

# Update symlink or reference
echo "See master_plan_v2.md for current plan" > .agents/plans/master_plan.md
```

**Option B: Version Within Existing Plan**
```markdown
# [Project] Master Plan

**Version**: 2.0 (Updated YYYY-MM-DD)  
**Previous Version**: [Link to archived v1]

## Deprecation Notice

~~Old section content~~ [DEPRECATED - see v2 below]

## NEW: Updated Section

[New content]

### Migration Notes
- [What changed from v1]
- [How to transition]
```

**Always**:
- Document why the change was needed
- Reference the original plan
- Notify human of significant restructuring
- Commit with descriptive message

---

## Critical Rules

### Rule 1: Always Check CURRENT_FOCUS First

**Command**:
```bash
cat .agents/CURRENT_FOCUS.md
```

**Never** start work without doing this.

**Why**: It shows:
- What's currently being worked on
- What's blocked
- What the priorities are
- Any human action items needed

**Consequence of breaking**: May work on wrong task, duplicate effort, or miss blockers

---

### Rule 2: Update Status Immediately

**Timeline**:
- Start work → Change to 🔄 within 1 minute
- Complete work → Change to ✅ within 1 minute
- Become blocked → Change to 🚫 within 1 minute

**Required fields**:
```markdown
**Status**: ✅ Complete (2026-01-30 14:30)
**Time Taken**: 45 minutes
```

**Why**: Maintains accurate project state

**Consequence of breaking**: Other agents/humans work from stale information

---

### Rule 3: Definition of Done is Law

A task is **NOT** complete until:
- [ ] All Definition of Done checkboxes are checked
- [ ] Verification commands have been run successfully
- [ ] Files are documented in "Files Created/Modified"
- [ ] No errors or warnings
- [ ] Status updated to ✅ with timestamp

**Why**: Prevents "almost done" work and ensures quality

**Consequence of breaking**: Incomplete work, bugs, missing files

---

### Rule 4: Human Tasks Block AI Tasks

**Process when encountering human task**:
1. Read task details in current_tasks.md
2. Mark human task as ⬜ (not started)
3. Add to "Human Action Items" in CURRENT_FOCUS.md
4. Document clearly what human needs to do
5. Find next unblocked AI task
6. **NEVER** attempt human tasks yourself

**Why**: Humans have credentials, permissions, and judgment needed

**Consequence of breaking**: Security violations, incorrect setup, broken state

---

### Rule 5: Commit Plan Changes

**When to commit**:
- After marking any task complete ✅
- After adding amendments
- After archiving a phase
- End of each work session

**Commit message format**:
```bash
git commit -m "docs: [phase] [action] - [brief description]"

# Examples:
git commit -m "docs: Phase 1 - mark AI 1.1 complete, start AI 1.2"
git commit -m "docs: Phase 2 - add amendment for GCS module config"
git commit -m "docs: archive Phase 0, start Phase 1"
git commit -m "docs: mark Human 1.1-1.3 complete, unblock AI 2.1"
```

**Why**: Maintains history, allows rollback, shows progress

**Consequence of breaking**: Lost work, confusion about state, no audit trail

---

## Error Handling

### Error: Can't Find CURRENT_FOCUS.md

**Symptoms**: File doesn't exist or wrong location

**Diagnosis**:
```bash
find . -name "CURRENT_FOCUS.md" 2>/dev/null
ls -la .agents/
```

**Resolution**:
1. Check if you're in the right directory
2. Look for `.agents/CURRENT_FOCUS.md`
3. If missing, ask human: "I can't find CURRENT_FOCUS.md. Should I create it using the template?"

**Prevention**: Always check file exists before reading

---

### Error: Task Has No Definition of Done

**Symptoms**: Task is vague, unclear when it's complete

**Example**:
```markdown
#### AI 1.1: Set up database ⬜
# BAD - no Definition of Done
```

**Resolution**:
1. Add Definition of Done section with 3-5 checkboxes
2. Make each criterion specific and verifiable
3. Add verification commands

**Example fix**:
```markdown
#### AI 1.1: Create database schema ⬜

**Definition of Done**:
- [ ] `schema.sql` created with tables: users, posts, comments
- [ ] Foreign keys defined between tables
- [ ] Migration script runs without errors
- [ ] `SELECT * FROM users` returns empty result (verifies connection)

**Verification**:
```bash
psql -d mydb -f schema.sql
psql -d mydb -c "SELECT * FROM users"
```
```

---

### Error: Dependencies Not Documented

**Symptoms**: Don't know what blocks a task, unclear order

**Resolution**:
1. Check CURRENT_FOCUS.md "Blocked" section
2. Read task description carefully for "Blocked By" field
3. If still unclear, ask human: "What are the prerequisites for [task]?"
4. Document the answer in the task's "Blocked By" field

---

### Error: Phase Taking Too Long

**Symptoms**: Phase stuck in 🔄 status for >7 days

**Resolution**:
1. Review CURRENT_FOCUS.md for blockers
2. Add note to phase header:
   ```markdown
   **Status**: 🔄 In Progress (STALE - in progress >7 days)
   ```
3. Analyze why it's stuck:
   - Dependencies not unblocked?
   - Tasks underestimated?
   - Blocked waiting for human?
4. Escalate to human with specific issues

---

### Error: Permission Denied Creating Files

**Symptoms**: Can't create .agents/ directory or files

**Resolution**:
1. Check current directory: `pwd`
2. Check permissions: `ls -la`
3. Ask human: "I need write permissions to create .agents/ directory. Can you grant access?"

**Alternative**: Create in /tmp first, then ask human to move

---

## Templates Quick Reference

### Task Template

```markdown
#### [AI/Human] X.Y: [Task Name] [STATUS]

**Status**: [STATUS] (YYYY-MM-DD HH:MM)
**Estimated Time**: XX minutes
**Priority**: High/Medium/Low
**Blocked By**: [Dependencies or "None"]

**Definition of Done**:
- [ ] Criterion 1 (specific and measurable)
- [ ] Criterion 2 (specific and measurable)
- [ ] Criterion 3 (specific and measurable)

**Steps**:
1. Step one with command
2. Step two with command
3. Step three with command

**Verification**:
```bash
# Command to verify
# Expected output
```

**Files Changed**:
- `path/to/file` (new/modified/deleted)

**Notes**:
[Special considerations]

**Amendments**:
- **YYYY-MM-DD**: [Description of change]
```

### Amendment Template

```markdown
### Amendment N: [Brief description]
**Date**: YYYY-MM-DD
**Reason**: [Why change was needed]
**Impact**: [What this affects]
**Changes Made**:
- [Change 1]
- [Change 2]
**Updated by**: AI/Human
```

### Archive Header Template

```markdown
# Phase X: [Name] [ARCHIVED]

**Status**: ✅ Complete
**Completed**: YYYY-MM-DD
**Archived**: YYYY-MM-DD
**Total Duration**: X days

## Summary
[What was accomplished]

## Key Deliverables
- [Item 1]
- [Item 2]

## Metrics
- Tasks completed: X/Y
- Estimated vs actual time: Xh vs Yh

## Lessons Learned
### What Worked Well
- [Success 1]

### What Could Be Improved
- [Improvement 1]

### Key Insights
- [Insight 1]

## Files Created
- `path/to/file` - [Purpose]
```

### CURRENT_FOCUS Template

```markdown
# Current Focus

**Last Updated**: YYYY-MM-DD HH:MM
**Active Phase**: Phase X - [Name]
**Next Milestone**: [What's coming]

## What's Happening Now

### In Progress 🔄
| Task | Phase | Assigned | Started |
|------|-------|----------|---------|
| [Task] | Phase X | AI/Human | Date |

### Blocked 🚫
| Task | Blocked By | ETA |
|------|------------|-----|
| [Task] | [By whom] | [When] |

### Ready to Start ⬜
| Task | Phase | Priority |
|------|-------|----------|
| [Task] | Phase X | High/Low |

## Today's Goal
1. [ ] Task 1
2. [ ] Task 2

## Phase Progress
| Phase | Status | Progress |
|-------|--------|----------|
| Phase 0 | ✅ Complete | 100% |
| Phase 1 | 🔄 In Progress | X% |

## Human Action Items
### Current: Phase X
[Table of tasks]

### Upcoming: Phase Y
[Table of tasks]

## Quick Links
- [Current Tasks](../todos/current_tasks.md)
- [Master Plan](../plans/master_plan.md)
```

---

## Success Metrics

A well-managed project using this system should have:

✅ **Documentation Quality**
- CURRENT_FOCUS updated daily
- All tasks have Definition of Done
- Clear dependencies documented
- Human action items visible

✅ **Timeliness**
- Timestamps on all status changes
- Tasks completed within estimated time ±20%
- Phases completed on target dates
- No tasks stuck in 🔄 >7 days

✅ **Organization**
- Completed phases archived
- Clear file structure
- Consistent naming conventions
- Regular commits

✅ **Communication**
- Blockers clearly documented
- Amendments explained
- Human notified of blockers
- Progress visible at a glance

---

## Quick Start Checklist

When starting a new project, use this checklist:

- [ ] Create .agents/ directory structure
- [ ] Create plans/master_plan.md with all 7 sections
- [ ] Create todos/current_tasks.md with Phase 0/1 tasks
- [ ] Create CURRENT_FOCUS.md with today's priorities
- [ ] Create .agents/README.md documenting the structure
- [ ] Add all files to git
- [ ] Commit: "docs: initialize project planning structure"
- [ ] Read CURRENT_FOCUS.md to confirm structure

**Time estimate**: 45-90 minutes for initial setup

---

## Version History

- **v1.0** (2026-01-30): Initial comprehensive guide

---

*For concise reference, see project memory: agent_planning_workflow*
