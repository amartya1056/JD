"""JobGuard — fake job posting detection core library.

This package holds every piece of logic shared between the training pipeline
(`scripts/train.py`) and the serving layer (`backend/app`). Keeping it in one
place guarantees train/serve parity: the same preprocessing and feature code
runs at fit time and at inference time.
"""

__version__ = "1.0.0"
