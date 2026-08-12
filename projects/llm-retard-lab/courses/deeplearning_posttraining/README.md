# Fine-Tuning and Reinforcement Learning for LLMs

Course sandbox for deeplearning.ai's [Fine-Tuning and RL for LLMs](https://learn.deeplearning.ai/courses/fine-tuning-and-reinforcement-learning-for-llms-intro-to-post-training) course. Took the course scaffolding and applied it to my own domain — retail customer service — instead of stopping at the notebooks.

## What I Built

End-to-end post-training artifact: synthetic data → LoRA SFT → graded eval comparing base vs. fine-tuned. 1/3 of a LinkedIn series; the eval chart is the anchor.

- **Domain**: Retail customer service (deferred: coding/MBPP)
- **Base model**: Qwen 1.5B
- **Method**: LoRA SFT on cold-start CoT samples
- **Iterations**: 3 (`adapter_qwen_cot` → `_cot_2` → `_cot_full`)
- **Result**: Fine-tuned 1.5B beats prompted base on the same retail CS rubric

## Notebooks

| Notebook | Purpose |
| ---------- | --------- |
| `notebooks/data_collection.ipynb` | Pydantic schemas for 5 stages (ColdStartCoTSFT, GeneralSFT, DPO, RewardModel, GRPO) + `validate_jsonl()` + OpenRouter synthetic generator |
| `notebooks/sft_lora_qwen.ipynb` | Local SFT runner (LoRA) |
| `notebooks/sft_lora_qwen_runpod.ipynb` | RunPod GPU run |
| `notebooks/evals.ipynb` | Base vs. fine-tuned grading on retail CS rubric |

## Artifacts

```
data/01_cold_start_cot_sft/
  data.jsonl          # 496 training samples, 940K
  eval.jsonl          # 30 held-out samples
  failures.jsonl
models/
  adapter_qwen_cot_full/    # final LoRA adapter
  adapter_qwen_cot_2/       # iter 2
  adapter_qwen_cot/         # iter 1
evals/
  accuracy_chart.png        # base vs fine-tuned — the LinkedIn anchor
  base.jsonl
  base_scored.jsonl
  finetuned.jsonl
  finetuned_scored.jsonl
```

## Schema Stages (from course)

1. ColdStartCoTSFT — reason + answer, retail CS
2. GeneralSFT — short-form answers
3. DPO — preference pairs
4. RewardModel — graded rubric
5. GRPO — rule-based rewards

Only stage 1 was run end-to-end for this artifact; 2–5 are scaffolded for later.

## Next

Post 2/3: MLOps primitive (registry / eval-CI gate / retrain trigger) — picked by who reacts to 1/3.
Post 3/3: serving on vLLM with latency/throughput chart.

## Using the Shared Library

```python
from llm_training.data.tokenizer import Tokenizer
from llm_training.models.gpt2 import GPT2Model, GPT2Config
from llm_training.training.dpo import DPOTrainer
```

```bash
cd ../..  # project root
pip install -e .
```

