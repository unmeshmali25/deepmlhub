# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python (posttraining)
#     language: python
#     name: posttraining
# ---

# %%
from importlib import reload
import sys
sys.path.append('../src/')

# %%
import llm_training.config as config_module

# %%
reload(config_module)

# %%
from pathlib import Path
config_path = Path("../configs/config.yaml")


# %%
src_train_config = config_module.load_train_config(config_path)

# %%
from dataclasses import asdict
for name, value in asdict(src_train_config).items():
    print(f"{name}: {value!r}")

# %%
assert src_train_config.model_name == "Qwen/Qwen2.5-1.5B"
assert src_train_config.lora_r == 16
assert src_train_config.lora_alpha == 32
assert src_train_config.lora_dropout == 0.05
assert src_train_config.learning_rate == 0.0002
assert src_train_config.data_path.exists()
assert isinstance(src_train_config.lora_target_modules, tuple)

print("Source configuration loader passed validation.")

# %%
