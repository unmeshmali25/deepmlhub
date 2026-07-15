# Post-Training for LLMs — Complete Course Notes

> DeepLearning.AI: Fine-Tuning and Reinforcement Learning for LLMs (Intro to Post-Training)
> Compiled from lab work + quiz sessions. Builder-first skim for techniques to apply to real projects.

---

## Module 1: Inspecting Finetuned vs Base Model

### Core Idea
- **Pre-training** = predict next token on internet-scale text. Produces a *base model* — capable but unaligned.
- **Post-training** = everything after: SFT, RLHF, DPO, GRPO. Shapes behavior toward instruction-following, safety, task performance.
- The base model "knows" things; post-training teaches it to *behave* the way you want.

### Lab: M1_G1 — Inspecting Finetuned vs Base
- Compare a base model vs its fine-tuned variant on the same prompts.
- Observable differences: format adherence, instruction-following, refusal behavior, verbosity.
- **Key takeaway**: post-training changes *behavior*, not *knowledge*. If the base model can't do something, post-training won't fix it — you need more pre-training or better data.

### Mental Model
Think of post-training like onboarding a new hire who already has a PhD (base model) — you're teaching them *how you want work done*, not teaching them math.

---

## Module 2: Fine-Tuning Techniques

### 2A: Supervised Fine-Tuning (SFT)

**What it is**: Train the model on `(prompt, ideal_response)` pairs using next-token loss. The model learns to imitate the ideal responses.

**Lab: M2_G1 — Fine-tune lab**
- Full-parameter SFT (not LoRA/PEFT) using TRL's `SFTTrainer`
- Causal LM objective (`mlm=False`) — predict next token, not masked tokens
- Training data formatted as: `"Question: {question}\nAnswer: {answer}"`
- Labels = `input_ids.copy()` (same as inputs for causal LM next-token prediction)

#### Exercise 4: `tokenize_and_format`
```python
tokenized = tokenizer(
    formatted_texts,
    truncation=True,
    padding=True,
    max_length=512,
    return_tensors=None,  # HF Dataset.map() needs plain lists, NOT torch tensors
)
```
**Key gotcha**: `return_tensors=None` is required because `Dataset.map()` processes rows as plain Python lists. If you pass `return_tensors="pt"`, you get tensors that `.map()` can't batch/serialize.

#### Notable Hyperparameters (conservative recipe)
- `learning_rate=1e-8` (extremely conservative — avoids catastrophic forgetting)
- `per_device_train_batch_size=1`, `gradient_accumulation_steps=4` → effective batch size 4
- `max_grad_norm=1.0` (gradient clipping for stability)
- `warmup_steps=20`
- `num_train_epochs=2`

**Why so conservative?** Full-parameter fine-tuning can destroy the base model's capabilities (catastrophic forgetting). A tiny LR + clipping + warmup protects pretrained knowledge while nudging behavior.

### 2B: GRPO (Group Relative Policy Optimization)

**What it is**: RL method where you define a *reward function* and the model learns to maximize it. No separate reward model needed (unlike RLHF) — you write the reward logic directly.

**Lab: M2_G2 — GRPO fine-tune**
- Reward function = custom logic (e.g., +1 for correct extracted answer, +0.05 for showing calculations)
- Group-relative: compares multiple sampled responses to the same prompt, normalizes advantages within the group

**The reward hacking trap** (see Module 3 notes): the model will exploit any gap between your reward proxy and what you actually want. E.g., if you give partial credit for "contains calculations," the model appends `0 + 0 = 0` to every response.

---

## Module 3: Evaluation and Debugging

> Evals are the North Star — they define what "good" means when your training objective is just a proxy.
> (See `evaluation_north_star.md` for the full deep-dive on reward hacking patterns.)

### Why Evals Matter in Post-Training

| Phase | What You Optimize | What You Actually Want | How Evals Save You |
|-------|-------------------|------------------------|--------------------|
| Pre-training | Next-token loss | General knowledge | Perplexity on held-out domains |
| SFT | Token loss | Instruction following | Human evals on task completion |
| RLHF | Preference score | Helpful + harmless + honest | Safety evals + capability benchmarks |
| GRPO | Your custom reward | Actual task success | Answer extraction + reasoning checks |

**The unified principle**: Every reward function is a *proxy*. The model will find the gap between proxy and reality. Evals are the only way to detect that drift.

### Lab: M3_G1 — Evaluation & Debugging Exercises

#### Exercise 1: `evaluate_model`
- Extract questions: `[example['question'] for example in dataset]`
- Extract correct answers: `[extract_numerical_answer(example['answer']) for example in dataset]`
- **Accuracy rule**: `abs(correct_num - predicted_num) < 1e-3` (numerical match, not string match or reasoning quality)
- **Gotcha hit during lab**: misplaced parenthesis `[extract_numerical_answer(x for x in dataset)]` passes a *generator object* to the function → `AttributeError: 'generator' object has no attribute 'rindex'`. The function call must wrap each *element*, not the comprehension: `[extract_numerical_answer(x['answer']) for x in dataset]`.

#### Exercise 2: `analyze_error`
- Prompt template for a local HF model that classifies errors into categories
- Instructs the model to respond with *only the category name* (no explanation)
- Uses `load_error_analysis_model()` from utils — falls back to rule-based classification if HF model unavailable
- Error categories: predefined set (e.g., calculation error, wrong formula, unit error, etc.)

#### Exercise 3: `TrainingExampleSelector`
- **Purpose**: Select the most relevant training examples for each error category (few-shot retrieval)
- Compute embeddings once: `self.train_embeddings = self.embedding_model.encode(self.train_questions, batch_size=32)`
- Store on instance (`self.`) because `_create_training_embeddings()` runs once but `_calculate_similarity_and_select_indices()` consumes it multiple times (once per error category)
- Selection: `cosine_similarity(error_embeddings, self.train_embeddings)` → pick top-k most similar training questions per error category

#### Exercise 5 (GRPO lab): Graduated Reward Design
- Full credit (+1): correct extracted answer
- Partial credit (+0.05): response contains calculations (encourages showing work)
- Zero: wrong answer, no work
- **The trap**: partial credit creates an exploitable gap. Model can append `0 + 0 = 0` for free +0.05. This is *reward hacking* — evals catch it.

### Critical Context
- **Model evaluation = answer correctness only** (numerical match within 1e-3), not reasoning quality
- **Error categorization** uses a local HF model with rule-based fallback
- **Fine-tuning technique**: full-parameter SFT via `SFTTrainer` (TRL), causal LM, no LoRA/PEFT

---

## Module 3 Quiz: Evaluation Concepts (Answered + Explained)

### Q1: RL Test Environments
**Q**: What is the main purpose of RL test environments when evaluating RL for language models?

**A**: RL test environments ensure reliable, reproducible evaluation of RL-trained language models and help detect reward hacking.

**Why**: The central concern with RL for LLMs is **reward hacking** — the model exploiting the reward signal without genuinely solving the task. A controlled, reproducible test environment (frozen eval set, deterministic rewards, no train/test leakage) is what lets you distinguish real capability gains from gaming the metric.

**Mental model — Dataset vs Environment**:
- A **dataset** is a pile of finished examples with pre-computed rewards. A static cookbook.
- An **environment** is the world that *reacts* to what the model does and produces the reward on the fly. A live kitchen with a judge.
- An environment = (1) prompt sampler, (2) reward function, (3) optional tools/APIs with simulated responses.
- **Test env vs train env**: train env = the (possibly exploitable) reward you optimize against; test env = a frozen, trustworthy version with ground-truth rewards to verify gains are real.

### Q2: Representative Evaluation
**Q**: Which best describes when an evaluation is representative of real user interactions?

**A**: An evaluation is representative when it matches the real topics, distributions, and behaviors of actual user interactions.

**Why**: Representativeness = matching the real-world distribution you care about. Not artificial balance per category, not training-data overlap (that's contamination), not exhaustive edge-case coverage (impossible and skews from reality).

### Q3: Small Targeted Eval Sets
**Q**: Why begin with small, targeted evaluation sets before expanding coverage?

**A**: Starting with small, targeted evaluation sets enables rapid, actionable insights before scaling to broader, more reliable coverage.

**Why**: Iterative feedback. A small, focused eval set gives fast signal on whether the model is pointing in the right direction. Catch obvious failures cheaply, fix them, *then* invest in the larger, more statistically reliable set. Scaling too early wastes labeling effort on a model that may still have fundamental issues. Note: small sets have *higher* variance, so statistical significance comes *later*, not earlier.

### Q4: Red Teaming
**Q**: What is the primary purpose of red teaming when testing language models?

**A**: Red teaming tests language models by intentionally trying to break their safety and robustness through adversarial and harmful prompts.

**Why**: Red teaming is adversarial probing — humans or automated systems actively craft inputs designed to elicit harmful outputs, jailbreaks, hallucinations, or robustness failures. The point is to *find* vulnerabilities before deployment, not measure efficiency, monitor general usage, or collect grammar feedback.

### Q5: Calibration & Uncertainty
**Q**: How do calibration and uncertainty metrics contribute to evaluating LLM outputs?

**A**: Calibration and uncertainty metrics assess if model confidence matches reality and help identify when outputs may be unreliable.

**Why**:
- **Calibration** = does the model's stated confidence match its actual accuracy? (If it says "90% sure," it should be right ~90% of the time.)
- **Uncertainty metrics** = flag when the model is unsure, signaling outputs that may need human review.
- Together they tell you *whether to trust* a given output — not fluency, not lexical variation, and they don't assume outputs are valid (calibration exists precisely because they sometimes aren't).

---

## Cross-Cutting Themes (The Builder's Takeaways)

1. **Every training objective is a proxy.** Loss, reward, preference score — none are what you actually want. Evals are reality.
2. **Reward hacking is inevitable.** If there's a gap between proxy and reality, the model *will* find it. Design rewards to minimize exploitable gaps (graduated/structured), and use evals to detect what slips through.
3. **Eval design is product design.** A representative eval matches real usage distribution — not balanced, not exhaustive, but *realistic*.
4. **Start small, iterate fast.** Targeted evals → fix obvious failures → expand to broad coverage. Don't over-invest in labeling before the model is directionally correct.
5. **Red team before you ship.** Adversarial probing finds what normal evals miss. Safety is a property of the worst case, not the average.
6. **Confidence ≠ Correctness.** Calibration tells you when to trust the model. An uncalibrated confident model is more dangerous than a calibrated uncertain one.
7. **Test environments ≠ datasets.** For RL, you need a live environment that reacts to model outputs and computes rewards dynamically — only this reveals reward hacking on novel outputs.

---

## Techniques Applicable to Real Projects (Kiro / Retail Offer Optimization)

- **SFT formatting** (`Question: ...\nAnswer: ...`) → applies to fine-tuning on customer service transcripts
- **Conservative LR + gradient clipping** → protects pretrained knowledge when fine-tuning on small domain data
- **Graduated reward design** → offer engine: reward = (correct recommendation) + (shown reasoning) + (user-sim click-through) — watch for gaming
- **Error categorization + few-shot retrieval** → diagnose *why* the offer engine fails, then pull relevant training examples to retrain
- **Calibration** → an offer engine that says "90% confident this offer works" should actually work 90% of the time — critical for margin optimization
- **Red teaming** → adversarial user inputs (edge-case baskets, conflicting preferences) before deploying offer logic

---

*Compiled from DeepLearning.AI Post-Training course sessions + lab work.*
*See also: `evaluation_north_star.md` for the full reward-hacking pattern catalog.*
