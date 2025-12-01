# src/__init__.py
from __future__ import annotations

import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*Sparse CSR tensor support is in beta state.*",
    category=UserWarning,
)

# Public API re-exports
from .types import *
from .utils import *
from .kernels import *
from .algorithm import *

from .model_api import Model, Parameters