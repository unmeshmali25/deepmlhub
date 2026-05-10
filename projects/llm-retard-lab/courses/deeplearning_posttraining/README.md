# Fine-Tuning and Reinforcement Learning for LLMs

Course-specific sandbox for deeplearning.ai's [Fine-Tuning and Reinforcement Learning for LLMs](https://learn.deeplearning.ai/courses/fine-tuning-and-reinforcement-learning-for-llms-intro-to-post-training) course.

## Structure

- `notebooks/` — Lesson notebooks and labs (added as you go)
- `assignments/` — Course assignment stubs (added as you go)
- `experiments/` — Your own hyperparameter sweeps and comparisons
- `data/` — Course datasets (Alpaca, Anthropic HH, etc.)

## Using the Shared Library

Import from the shared package in any notebook or script:

```python
from llm_training.data.tokenizer import Tokenizer
from llm_training.models.gpt2 import GPT2Model, GPT2Config
from llm_training.training.dpo import DPOTrainer
```

Install the package in editable mode:

```bash
cd ../..  # project root
pip install -e .
```
