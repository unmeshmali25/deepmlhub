"""
Notebook Portability Utilities
==============================

WHY THIS MODULE EXISTS
----------------------
Notebooks are environment-agnostic documents that run in environment-specific
kernels. This creates a fundamental portability problem: a notebook written on
your M1 Pro Mac will fail on Google Colab, and a notebook tuned for Colab's
T4 GPU will crash on your laptop's CPU.

THE PROBLEM WITH NAIVE NOTEBOOKS
--------------------------------
Without this module, every notebook needs repetitive, error-prone bootstrap code:

    import sys, os
    # Was the repo already cloned? Is it the right path?
    if not os.path.exists('deepmlhub'):
        !git clone ...
    # Is src/ on the path? Depends on how the notebook was opened...
    sys.path.insert(0, 'src')  # Wrong if cwd is courses/
    # What device am I on? cuda? mps? cpu?
    # Should I use mixed precision? Only CUDA supports torch.cuda.amp reliably.
    # Is the data downloaded? If not, the first training cell crashes.

This boilerplate distracts from learning. Worse, it breaks silently when you
move between environments.

THE SOLUTION: ONE CALL PER NOTEBOOK
-----------------------------------
Every notebook starts with a single cell:

    from llm_training.utils.notebook import setup_notebook
    config = setup_notebook("configs/gpt2_124m_pretrain.yaml")

This handles: environment detection, path setup, config patching, data checking,
and (if needed) cloud authentication.

THE SIX PORTABILITY LAYERS
--------------------------
1. ENVIRONMENT DETECTION
   We need to know if we're on local Mac, Google Colab, or Lambda AI.
   Each has different file system layouts, package availability, and auth
   requirements. We detect this at runtime using module availability and
   filesystem heuristics, not by asking the user.

2. PACKAGE BOOTSTRAP
   The `llm_training` package lives at the repo root under `src/llm_training/`.
   On a fresh Colab/Lambda VM, the repo doesn't exist and the package isn't
   installed. We need to clone, install, and add `src/` to `sys.path`.
   On local, `pip install -e .` may have already been run, but `sys.path`
   still needs `src/` if the notebook is in `courses/.../`.

3. DEVICE & PRECISION AUTO-DETECTION
   Configs should be hardware-agnostic. The same `gpt2_124m_pretrain.yaml`
   should work on M1 (MPS), Colab T4 (CUDA), and Lambda A100 (CUDA).
   We auto-detect the best available device and disable mixed precision
   on non-CUDA backends where torch.cuda.amp does not exist.

4. DATA PROVISIONING
   Relative paths like `data/shakespeare/train.bin` work locally because
   you ran `prepare.py` once. On a fresh cloud VM, the data directory is
   empty. We check for data existence and auto-run prepare scripts.

5. CHECKPOINT PERSISTENCE
   Colab and Lambda VMs are ephemeral. A 2-hour training run is wasted
   if checkpoints only live in the local `outputs/` directory. On cloud
   environments, we should save to GCS or Drive. On local, disk is fine.

6. AUTHENTICATION
   Google Cloud Storage requires credentials. On your Mac, you have
   `gcloud` configured. On Colab, you need OAuth via a browser popup.
   On Lambda, you might have a service account key. We handle the
   appropriate auth method per environment.

DESIGN DECISION: NO NOTEBOOK-SPECIFIC LOGIC IN TRAINING CODE
------------------------------------------------------------
The training scripts (`pretrain.py`, `sft.py`, `dpo.py`) already call
`get_default_device()` and do NOT read `config["device"]["type"]`.
This means we can safely remove the `device:` section from configs.
The training code stays clean and the notebook layer handles all
environment-specific overrides.

WHY NOT JUST USE REQUIREMENTS.TXT?
----------------------------------
`requirements.txt` lists package versions but does not:
- Clone the repository (needed on fresh VMs)
- Add `src/` to `sys.path` (needed for local development)
- Install the package in editable mode (needed for `import llm_training`)
- Handle Colab's `%pip` vs `!pip` distinction

WHY NOT AUTO-DETECT BATCH SIZE?
-------------------------------
Auto-scaling batch size based on available GPU memory is complex
(requires forward passes, OOM catching, binary search). Instead, we
provide a simple override hook and let you tune manually per machine.
The config holds the "target" batch size; the notebook can override it.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import yaml

from llm_training.utils.device import get_device, is_cuda


def detect_environment() -> str:
    """
    Detect the execution environment.

    WHY: Different environments require different setup steps:
    - Colab needs `%pip` magic and OAuth
    - Lambda needs `!pip` and may have pre-configured service accounts
    - Local may already have the repo cloned and package installed

    HOW: We use a cascade of heuristics:
    1. Check if `google.colab` is already imported (Colab pre-imports it)
    2. Check for Lambda-specific filesystem markers
    3. Default to "local"
    """
    if "google.colab" in sys.modules:
        return "colab"

    # Lambda AI cloud instances often have /opt/lambda or specific env vars
    if os.path.exists("/opt/lambda") or os.environ.get("LAMBDA_CLOUD"):
        return "lambda"

    return "local"


def setup_paths(env: str) -> Path:
    """
    Ensure `sys.path` contains the repo `src/` directory.

    WHY: Notebooks in `courses/` are nested 2-3 levels deep relative to `src/`.
    `import llm_training` fails unless Python knows to look in the right place.

    HOW: We walk up from the current file location until we find `src/llm_training/`.
    If running in Colab/Lambda, we first clone the repo. Then we insert the
    absolute path to `src/` at position 0 of `sys.path`.

    Note: We use the notebook's __file__ (if available) or cwd as the anchor.
    """
    # In notebooks, __file__ is often not defined. Fall back to cwd.
    cwd = Path.cwd()

    # If we're deep inside courses/, walk up to the project root
    if "courses" in str(cwd) or "experiments" in str(cwd):
        # Walk up until we find src/llm_training or hit the filesystem root
        root = cwd
        while root != root.parent:
            if (root / "src" / "llm_training").exists():
                break
            root = root.parent
    else:
        root = cwd

    src_path = root / "src"
    if str(src_path) not in sys.path:
        # Insert at front so our package shadows any accidental pip installs
        sys.path.insert(0, str(src_path))

    return root


def clone_and_install(env: str, root: Path, repo_url: str = "https://github.com/unmeshmali/deepmlhub.git") -> Path:
    """
    On fresh cloud VMs, clone the repo and pip-install the package.

    WHY: Colab and Lambda start with a blank filesystem. The notebook code
    imports `llm_training`, but the package doesn't exist yet. We need to:
    1. Clone the repo from GitHub
    2. cd into the project
    3. `pip install -e .` so `import llm_training` works

    HOW: Colab prefers `%pip` (IPython magic) which installs into the current
    kernel without restart. Other environments use `subprocess.run` with the
    system Python.

    Args:
        env: Environment name from detect_environment()
        root: Project root path (may change after clone)
        repo_url: GitHub URL of the repository

    Returns:
        Updated project root path after potential clone
    """
    project_dir = root / "projects" / "llm-retard-lab"

    # Already cloned?
    if project_dir.exists():
        return project_dir

    if env == "colab":
        # IPython magic works in notebook cells; here we simulate it
        print(f"[notebook] Cloning {repo_url} into {root}...")
        subprocess.run(["git", "clone", repo_url, str(root / "deepmlhub")], check=True)
        project_dir = root / "deepmlhub" / "projects" / "llm-retard-lab"
        os.chdir(project_dir)
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".", "-q"], check=True)
    elif env == "lambda":
        print(f"[notebook] Cloning {repo_url}...")
        subprocess.run(["git", "clone", repo_url, str(root / "deepmlhub")], check=True)
        project_dir = root / "deepmlhub" / "projects" / "llm-retard-lab"
        os.chdir(project_dir)
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
    else:
        # Local: assume repo is already cloned. Just ensure src/ is on path.
        pass

    return project_dir


def load_config_with_overrides(config_path: str) -> dict:
    """
    Load YAML config and patch hardware-dependent settings.

    WHY: A config committed to Git should work on every machine. Hardcoding
    `device.type: cuda` breaks on M1 Macs. Hardcoding `device.type: mps`
    breaks on Colab. The config should express *intent* (train GPT-2 124M);
    the runtime should express *capability* (CUDA, MPS, or CPU).

    WHAT WE PATCH:
    1. device.type → auto-detected best available backend
    2. device.mixed_precision → true only on CUDA (torch.cuda.amp)
    3. device.name → human-readable label for logging

    WHAT WE DO NOT PATCH (by design):
    - batch_size: We could auto-scale, but OOM recovery is fragile. Instead,
      the caller can override manually: `config['training']['batch_size'] = 8`
    - max_iters: This is a training budget, not a hardware concern.

    Args:
        config_path: Path to YAML config, relative to project root

    Returns:
        Config dict with hardware-aware overrides applied
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path.absolute()}")

    with open(path) as f:
        config = yaml.safe_load(f)

    # Auto-detect device
    detected = get_device()  # "cuda", "mps", or "cpu"

    # Ensure device section exists (configs may omit it now)
    if "device" not in config:
        config["device"] = {}

    config["device"]["type"] = detected
    config["device"]["name"] = _device_name(detected)

    # Mixed precision: only enable on CUDA where torch.cuda.amp is mature
    # MPS support for AMP is improving but still has edge cases in PyTorch 2.x
    # CPU AMP exists but is usually slower than full precision
    config["device"]["mixed_precision"] = is_cuda()

    # Log what we changed so the user isn't surprised
    print(f"[notebook] Config loaded from {path}")
    print(f"[notebook] Auto-detected device: {detected}")
    print(f"[notebook] Mixed precision: {config['device']['mixed_precision']}")

    return config


def _device_name(device: str) -> str:
    """Return a human-readable device label for logging."""
    if device == "cuda":
        import torch
        return f"cuda:{torch.cuda.get_device_name(0)}"
    if device == "mps":
        return "mps (Apple Silicon)"
    return "cpu"


def ensure_data(config: dict) -> None:
    """
    Verify that training data exists; run prepare script if missing.

    WHY: On your Mac, you ran `python data/shakespeare/prepare.py` once and
    forgot about it. On a fresh Colab VM, `data/shakespeare/train.bin` does
    not exist. If the training script starts first, it crashes with a cryptic
    FileNotFoundError deep in the DataLoader.

    HOW: We check the paths declared in the config. If a `.bin` or `.jsonl`
    is missing, we look for a `prepare.py` in the same data directory and
    run it. This is idempotent — safe to call every time.

    LIMITATION: This works for our built-in datasets (Shakespeare) where a
    prepare script exists. For external datasets (e.g., downloading Alpaca
    from HuggingFace), you still need manual setup.
    """
    data_cfg = config.get("data", {})
    checked = False

    for key in ["train_path", "val_path"]:
        path_str = data_cfg.get(key)
        if not path_str:
            continue

        path = Path(path_str)
        if path.exists():
            continue

        checked = True
        # Look for a prepare.py in the parent directory of the data file
        prepare_script = path.parent / "prepare.py"
        if prepare_script.exists():
            print(f"[notebook] Data missing: {path}")
            print(f"[notebook] Running prepare script: {prepare_script}")
            subprocess.run([sys.executable, str(prepare_script)], check=True)
        else:
            print(f"[notebook] WARNING: Data missing and no prepare.py found: {path}")

    if not checked:
        print("[notebook] All data paths verified.")


def setup_gcs_auth(env: str) -> None:
    """
    Authenticate with Google Cloud Storage if running in a cloud environment.

    WHY: Your checkpoint bucket is `gs://deepmlhub-voiceoffers-dvc/`. On your
    Mac, `gcloud` is already configured. On Colab, you need OAuth. On Lambda,
    you might have a service account key mounted. Without auth, checkpoint
    saves silently fail or raise `DefaultCredentialsError`.

    HOW PER ENVIRONMENT:
    - Colab: Use `google.colab.auth.authenticate_user()` which opens a popup.
    - Lambda: Check for `GOOGLE_APPLICATION_CREDENTIALS` env var.
    - Local: Assume `gcloud auth application-default login` was already run.

    This is a no-op if auth is already configured.
    """
    if env == "colab":
        try:
            from google.colab import auth as colab_auth
            print("[notebook] Authenticating with Google Cloud (Colab)...")
            colab_auth.authenticate_user()
            print("[notebook] GCS auth complete.")
        except Exception as e:
            print(f"[notebook] GCS auth skipped: {e}")
    elif env == "lambda":
        creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if creds:
            print(f"[notebook] Using GCS service account: {creds}")
        else:
            print("[notebook] WARNING: No GOOGLE_APPLICATION_CREDENTIALS found.")
    else:
        # Local: nothing to do; user manages gcloud CLI
        pass


def setup_notebook(
    config_path: str = "configs/gpt2_124m_pretrain.yaml",
    repo_url: str = "https://github.com/unmeshmali/deepmlhub.git",
    enable_gcs: bool = True,
) -> dict:
    """
    One-call notebook bootstrap.

    This is the PRIMARY ENTRYPOINT. Every notebook should start with:

        from llm_training.utils.notebook import setup_notebook
        config = setup_notebook("configs/gpt2_124m_pretrain.yaml")

    It composes all the individual utilities into the correct order:
    1. Detect environment (Colab / Lambda / Local)
    2. Set up import paths so `import llm_training` works
    3. Clone and install package on fresh cloud VMs
    4. Load config and patch hardware-dependent settings
    5. Ensure data files exist (run prepare.py if needed)
    6. Authenticate with GCS for checkpoint persistence (optional)

    Args:
        config_path: Relative path to YAML config from project root
        repo_url: GitHub URL for cloning on fresh VMs
        enable_gcs: Whether to trigger GCS auth in cloud environments

    Returns:
        Hardware-patched config dict ready for training
    """
    print("=" * 60)
    print("llm-retard-lab notebook bootstrap")
    print("=" * 60)

    env = detect_environment()
    print(f"[notebook] Environment detected: {env}")

    root = setup_paths(env)
    print(f"[notebook] Project root: {root}")

    project_dir = clone_and_install(env, root, repo_url)
    if project_dir != root:
        print(f"[notebook] Working directory: {project_dir}")
        # After clone, configs are relative to project_dir
        config_full_path = project_dir / config_path
    else:
        config_full_path = root / config_path

    # If the config path doesn't exist from cwd, try relative to project
    if not config_full_path.exists() and not Path(config_path).exists():
        # Maybe we changed directory; try absolute from project_dir
        config_full_path = project_dir / config_path

    # Load with hardware overrides
    config = load_config_with_overrides(str(config_full_path))

    # Ensure data is present
    ensure_data(config)

    # Cloud auth for checkpoints
    if enable_gcs:
        setup_gcs_auth(env)

    print("=" * 60)
    print("Notebook bootstrap complete. Happy training!")
    print("=" * 60)

    return config
