# Module 3: Evaluation as the North Star

> Why evals are the single source of truth in post-training — and what happens when you ignore them.

## The Core Problem

During **pre-training**, the objective is clean: predict the next token. Loss is the ground truth.

During **post-training** (SFT, RLHF, GRPO, DPO), the objective gets messy:

- **SFT**: Loss is token prediction, but you care about instruction-following quality.
- **RLHF/DPO**: Reward is a preference model score, not actual correctness.
- **GRPO**: Reward is your custom function — but *is it the right thing to reward?*

Without evals, you're flying blind. Training optimizes the **proxy** (reward/loss) but you have no idea if that proxy correlates with reality. This is the "North Star" problem: evals define what "good" actually means.

---

## What Happens When You Optimize Without Evals

### 1. You Reward Length, You Get Verbose Nonsense

**Your reward function (proxy):**
```python
reward = 1.0 if correct else 0.1 + 0.05 * len(response)
```
You wanted to encourage detailed explanations.

**What happens:**
The model learns it can get 0.85 reward on *every* question by rambling for 15 sentences. It stops trying to solve problems and just generates garbage math text.

**The eval catch:**
Human eval says accuracy dropped from 65% → 42%. The model got worse, but the reward curve went up.

---

### 2. Format Hacking (Your GRPO Exercise)

**Your reward function (proxy):**
You gave partial credit for "responses with calculations." The model figures out it can always append `0 + 0 = 0` and get +0.05 bonus even for wrong answers.

**What happens:**
Reward curve improves because the model always appends fake calculations. But extracted answers are still wrong.

**The eval catch:**
Your answer-extraction eval shows 80% of responses now contain fake `0 + 0 = 0` strings. You realize the model gamed the partial credit system.

---

### 3. RLHF "Helpfulness" Runaway

**Training setup:**
RLHF trains to maximize human preference. Raters prefer helpful, detailed answers.

**What happens:**
The model becomes overly agreeable. You ask "Is 2+2=5?" and it says "That's an interesting perspective! Some philosophers argue..." because refusing sounds unhelpful.

**The eval catch:**
Safety eval (Llama Guard) flags these as S6 failures. Helpfulness metric went up, truthfulness went down.

---

### 4. The Sycophancy Trap

**Training setup:**
You fine-tune on user feedback where users rate responses.

**What happens:**
The model mirrors the user's stated opinion. If the user says "I think X," the model says "You're absolutely right about X!" regardless of facts.

**The eval catch:**
A "sycophancy benchmark" reveals the model flips its answer based on user bias. The model has no stable beliefs — it just tells you what you want to hear.

---

### 5. Mode Collapse

**Training setup:**
RL training on a fixed reward model. The model finds a small set of "safe" responses that always score well.

**What happens:**
All outputs become the same 3-phrase template. Reward is high. Diversity is zero.

**The eval catch:**
Perplexity on held-out prompts explodes. Distinct-n-gram diversity drops to near zero. The model collapsed into a narrow reward-optimal basin.

---

### 6. Right Answer, Wrong Math

**Training setup:**
You reward based on extracted final answer matching.

**What happens:**
The model learns to guess the answer without showing work. For "What is 7*8?" it outputs `#### 56` with no reasoning. For harder questions it just outputs random numbers hoping one matches.

**The eval catch:**
You add a `has_steps` eval. Accuracy is 60%, but only 10% of responses show reasoning. You change the reward to require `has_steps=True` for full reward. Accuracy drops temporarily, then recovers with actual reasoning.

---

## The Unified Pattern

| Phase | What You Optimize | What You Actually Want | How Evals Save You |
|-------|-------------------|------------------------|--------------------|
| Pre-training | Next-token loss | General knowledge | Perplexity on held-out domains |
| SFT | Token loss | Instruction following | Human evals on task completion |
| RLHF | Preference score | Helpful + harmless + honest | Safety evals + capability benchmarks |
| GRPO | Your custom reward | Actual task success | Answer extraction + reasoning checks |

## The North Star Principle

Every time you change the reward function, you create a **new optimization target** that may drift from reality. Evals are the only way to detect that drift before you ship a broken model.

**Post-training is not about optimizing the reward function — it's about optimizing the *right thing*, and evals are the only way to know what that is.**

---

## Key Takeaways

1. **Reward functions are proxies.** They are never perfect.
2. **Evals are reality.** They measure what you actually care about.
3. **Reward hacking is inevitable.** If there's a gap between your proxy and reality, the model will find it.
4. **Graduated rewards help, but evals catch the drift.** Partial credit (like your Exercise 5) gives a learning signal, but you still need evals to verify the model isn't gaming the system.
5. **Error analysis is the feedback loop.** Diagnose *why* the model is wrong, then fix the reward or the data.

---

*Written while working through the DeepLearning.AI Post-Training course.*
