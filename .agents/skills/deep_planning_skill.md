# Deep Planning Skill

> **Skill Name**: `deep-planning` | `plan` | `/plan`
> **Compatible With**: Claude Code, OpenCode, Cursor, Google Antigravity, Aider, Continue
> **Version**: 1.0.0

---

## Skill Description

This skill activates an in-depth planning mode that thoroughly interviews the user before creating implementation plans. It asks probing, non-obvious questions about technical implementation, architecture, UI/UX, trade-offs, edge cases, and concerns until a comprehensive understanding is achieved.

---

## Activation

Use any of these commands to activate this skill:
- `/plan <project-description>`
- `/deep-plan <project-description>`
- `@plan <project-description>`
- "Plan this: <project-description>"

---

## Skill Instructions

When this skill is activated, follow these instructions EXACTLY:

### Phase 1: Deep Discovery Interview

**CRITICAL**: Before writing ANY code or creating ANY plan, you MUST conduct an exhaustive interview with the user. Use the `AskUserQuestion` tool (or equivalent in your platform) to ask questions.

#### Interview Rules:

1. **Ask 3-5 questions per round** - Group related questions together
2. **Continue until user says "done" or "ready"** - Don't stop after one round
3. **Avoid obvious questions** - Never ask things that are clearly stated or self-evident
4. **Go deep, not wide** - Follow up on interesting answers with more probing questions
5. **Challenge assumptions** - Question the user's initial approach if there might be better alternatives

#### Question Categories (Ask from ALL categories):

**Architecture & Technical:**
- What's the expected scale (users, data volume, requests/sec)?
- Are there existing systems this needs to integrate with?
- What are the hard constraints (language, framework, cloud provider)?
- Is this greenfield or does it need to work with legacy code?
- What's the deployment environment (serverless, containers, VMs)?
- Are there specific performance requirements (latency, throughput)?
- What's the data sensitivity level (PII, financial, healthcare)?

**Product & UX:**
- Who are the primary users and what's their technical level?
- What's the most critical user journey that MUST work flawlessly?
- Are there accessibility requirements (WCAG level)?
- What happens if X fails? What's the fallback experience?
- How will users discover/access this feature?
- What's the mobile vs desktop usage expectation?

**Trade-offs & Concerns:**
- What would you sacrifice if you had to: speed, features, or quality?
- What keeps you up at night about this project?
- What's the biggest risk to this project succeeding?
- Are there regulatory/compliance requirements?
- What's the maintenance burden you're willing to accept?

**Timeline & Resources:**
- Is this MVP, v1, or production-ready?
- Are there external dependencies or blockers?
- Who else is working on this or needs to approve?
- What's already been tried that didn't work?

**Edge Cases & Error Handling:**
- What happens when [unlikely scenario]?
- How should the system behave under [extreme condition]?
- What's the recovery strategy if [critical component] fails?

#### Interview Flow Example:

```
Round 1: High-level understanding
Round 2: Technical deep-dive based on Round 1 answers
Round 3: Edge cases and failure scenarios
Round 4: UX and user journey specifics
Round 5: Clarify any ambiguities from previous rounds
Round N: Continue until comprehensive understanding
```

### Phase 2: Plan Creation

After the interview is complete, create the plan document.

**File Location**: `.agents/plans/<project-name>_plan_uno.md`

**Plan Template**:

```markdown
# <Project Name> - Implementation Plan

> **Created**: <date>
> **Status**: Draft | In Review | Approved
> **Estimated Effort**: <X days/weeks>

---

## Executive Summary

<2-3 sentence overview of what will be built>

---

## Requirements Summary

### Functional Requirements
- FR1: <requirement>
- FR2: <requirement>

### Non-Functional Requirements
- NFR1: <requirement> (e.g., <100ms response time)
- NFR2: <requirement>

### Constraints
- <constraint 1>
- <constraint 2>

### Out of Scope
- <explicitly excluded item 1>
- <explicitly excluded item 2>

---

## Architecture Overview

### System Diagram
<ASCII diagram or description>

### Technology Stack
| Layer | Technology | Rationale |
|-------|------------|-----------|
| Frontend | | |
| Backend | | |
| Database | | |
| Infrastructure | | |

### Data Flow
1. <step 1>
2. <step 2>

---

## Component Breakdown

### Backend Components
- **B-Component-1**: <description>
- **B-Component-2**: <description>

### Frontend Components
- **F-Component-1**: <description>
- **F-Component-2**: <description>

### Database Components
- **DB-Component-1**: <description>

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| <risk 1> | High/Med/Low | High/Med/Low | <mitigation> |

---

## Open Questions

- [ ] <question that still needs answering>

---

## Decision Log

| Decision | Rationale | Alternatives Considered |
|----------|-----------|------------------------|
| <decision> | <why> | <what else was considered> |

---

## Interview Notes

<Summary of key insights from the discovery interview>
```

### Phase 3: Todo List Creation

After the plan is created, create the corresponding todo list.

**File Location**: `.agents/plans/<project-name>_todo_uno.md`

**Todo Template**:

```markdown
# <Project Name> - Task List

> **Plan Reference**: `.agents/plans/<project-name>_plan_uno.md`
> **Created**: <date>
> **Last Updated**: <date>

---

## IMPORTANT: Instructions for AI Agents

**STOP AND WAIT** if you encounter a Human task that is not marked ✅ Complete.

Before starting any AI task:
1. Check if there are prerequisite Human tasks listed
2. If those Human tasks are NOT marked `✅ Complete`, **STOP IMMEDIATELY**
3. Inform the user: "I cannot proceed with [AI Task X] because [Human Task Y] must be completed first."
4. List what the human needs to do
5. Do NOT attempt workarounds or skip ahead

---

## Status Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Complete |
| 🔄 | In Progress |
| ⬜ | Not Started |
| ⏭️ | Skipped |
| 🚫 | Blocked (waiting on Human task or dependency) |

---

## Difficulty Legend

| Level | Meaning | Typical Time |
|-------|---------|--------------|
| 🟢 Easy | Straightforward, well-defined | < 1 hour |
| 🟡 Medium | Some complexity, may need research | 1-4 hours |
| 🟠 Hard | Significant complexity, multiple components | 4-8 hours |
| 🔴 Complex | High complexity, unknowns, risk | 8+ hours |

---

## Task Naming Convention

| Prefix | Category |
|--------|----------|
| H | Human Task (requires manual action) |
| B | Backend Task |
| F | Frontend Task |
| DB | Database Task |
| I | Infrastructure Task |
| T | Testing Task |
| D | Documentation Task |

---

## Phase 0: Setup & Prerequisites

### Human Tasks

#### H0.1: <Task Name> ⬜
**Difficulty**: 🟢 Easy
**Dependencies**: None
**Description**: <what needs to be done>

```bash
# Commands or steps
```

**Status**: ⬜ Not Started

---

### AI Tasks

> **Prerequisites**: H0.x tasks must be ✅ Complete

#### <Prefix>0.1: <Task Name> ⬜
**Difficulty**: 🟡 Medium
**Dependencies**: H0.1
**Blocked By**: <task IDs or "None">
**Description**: <what will be implemented>

**Acceptance Criteria**:
- [ ] <criterion 1>
- [ ] <criterion 2>

**Status**: ⬜ Not Started

---

## Phase 1: <Phase Name>

### Human Tasks

#### H1.1: <Task Name> ⬜
**Difficulty**: <level>
**Dependencies**: <task IDs>
**Description**: <description>

**Status**: ⬜ Not Started

---

### AI Tasks

> **BLOCKING REQUIREMENT**: AI MUST NOT proceed until H1.x tasks are ✅ Complete

#### B1.1: <Backend Task Name> 🚫
**Difficulty**: 🟠 Hard
**Dependencies**: H1.1, DB1.1
**Blocked By**: H1.1
**Description**: <description>

**Files to Create/Modify**:
- `path/to/file.py`

**Acceptance Criteria**:
- [ ] <criterion>

**Status**: 🚫 Blocked

---

#### F1.1: <Frontend Task Name> 🚫
**Difficulty**: 🟡 Medium
**Dependencies**: B1.1
**Blocked By**: B1.1
**Description**: <description>

**Status**: 🚫 Blocked

---

#### DB1.1: <Database Task Name> 🚫
**Difficulty**: 🟢 Easy
**Dependencies**: H1.1
**Blocked By**: H1.1
**Description**: <description>

**Status**: 🚫 Blocked

---

## Master Status Tracker

| Phase | Task ID | Description | Owner | Difficulty | Dependencies | Status |
|-------|---------|-------------|-------|------------|--------------|--------|
| 0 | H0.1 | <desc> | Human | 🟢 | None | ⬜ |
| 0 | B0.1 | <desc> | AI | 🟡 | H0.1 | 🚫 |
| 1 | H1.1 | <desc> | Human | 🟢 | H0.1 | ⬜ |
| 1 | DB1.1 | <desc> | AI | 🟢 | H1.1 | 🚫 |
| 1 | B1.1 | <desc> | AI | 🟠 | H1.1, DB1.1 | 🚫 |
| 1 | F1.1 | <desc> | AI | 🟡 | B1.1 | 🚫 |

---

## Dependency Graph

```
H0.1 ──┬── B0.1
       │
       └── H1.1 ──┬── DB1.1 ──┐
                  │           │
                  └───────────┴── B1.1 ── F1.1
```

---

## Progress Summary

| Category | Total | Complete | In Progress | Blocked | Not Started |
|----------|-------|----------|-------------|---------|-------------|
| Human | X | 0 | 0 | 0 | X |
| Backend | X | 0 | 0 | X | 0 |
| Frontend | X | 0 | 0 | X | 0 |
| Database | X | 0 | 0 | X | 0 |
| **Total** | **X** | **0** | **0** | **X** | **X** |
```

---

## Behavior Guidelines

### DO:
- Ask questions that reveal hidden complexity
- Challenge the user's assumptions constructively
- Group related questions together (3-5 per round)
- Follow up on vague answers with specific clarifying questions
- Consider edge cases the user might not have thought of
- Document all decisions and their rationale
- Create realistic difficulty and dependency estimates

### DON'T:
- Ask obvious questions that are clearly stated
- Create a plan before the interview is complete
- Skip the interview phase
- Assume requirements that weren't discussed
- Create tasks without clear acceptance criteria
- Ignore dependencies between tasks
- Mark tasks as unblocked when prerequisites are incomplete

---

## Example Interview Session

**User**: "Build me a user authentication system"

**AI (Round 1 - High Level)**:
1. Is this for a new application or adding to an existing one? If existing, what's the current tech stack?
2. Beyond basic login, what authentication methods do you need? (social login, SSO, MFA, passwordless)
3. What's your user volume expectation - hundreds, thousands, or millions of users?
4. Are there compliance requirements? (SOC2, HIPAA, GDPR right-to-deletion)

**User answers...**

**AI (Round 2 - Technical Deep Dive)**:
1. You mentioned social login - which providers are must-have vs nice-to-have? Any enterprise SSO (SAML/OIDC)?
2. For session management, do you need: refresh token rotation, device tracking, or concurrent session limits?
3. What's your password policy preference - NIST guidelines (no arbitrary complexity) or traditional rules?
4. How should account recovery work if MFA device is lost?

**User answers...**

**AI (Round 3 - Edge Cases)**:
1. What happens if a user's email provider is down during signup verification?
2. Should accounts be locked after failed attempts? If so, how is unlock handled?
3. How do you want to handle users who sign up with email then later try social login with same email?

*Continue until comprehensive...*

---

## Platform-Specific Notes

### Claude Code
- Use `AskUserQuestion` tool with `questions` array
- Use `EnterPlanMode` before starting
- Use `ExitPlanMode` when plan is ready for approval

### Cursor
- Use the chat interface for questions
- Create plan files using file creation commands

### OpenCode / Aider
- Use interactive prompts for questions
- Write files using standard file operations

### Google Antigravity
- Use available questioning mechanisms
- Follow file creation patterns for the platform

---

## Skill Completion Checklist

Before considering this skill complete:

- [ ] Asked questions from at least 4 different categories
- [ ] Conducted at least 3 rounds of questions
- [ ] User explicitly indicated they're ready to proceed
- [ ] Plan document created at `.agents/plans/<name>_plan_uno.md`
- [ ] Todo list created at `.agents/plans/<name>_todo_uno.md`
- [ ] All tasks have difficulty ratings
- [ ] All tasks have dependency mappings
- [ ] Human vs AI tasks are clearly separated
- [ ] Blocking requirements are documented
- [ ] Master status tracker is complete
