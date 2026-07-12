"""
conftest.py — Shared pytest configuration for DeepShield Audio test suite.

Sets up sys.path so that `from src.xxx import yyy` works correctly
regardless of where pytest is invoked from.
"""

import sys
import os
from pathlib import Path

# Ensure project root is on sys.path so `src.*` imports work
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Suppress TensorFlow verbose startup logs during testing
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
