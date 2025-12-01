#!/usr/bin/env python
# demo/demo_model_api.py

import os
import sys
import numpy as np
from scipy import sparse

# Add project root (the folder that contains src/) to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model_api import Model, Parameters


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Build LP from NumPy / SciPy arrays
    # ------------------------------------------------------------------
    # Maximize: 3 x1 + 5 x2
    # Subject to:
    #   1 x1 + 2 x2 <= 10
    #   3 x1 + 1 x2 <= 12
    #   x1, x2 >= 0

    A = sparse.csr_matrix(
        [
            [1.0, 2.0],
            [3.0, 1.0],
        ],
        dtype=np.float64,
    )

    # Row bounds on A x:
    AL = np.array([-np.inf, -np.inf], dtype=np.float64)
    AU = np.array([10.0, 12.0], dtype=np.float64)

    # Variable bounds:
    l = np.array([0.0, 0.0], dtype=np.float64)
    u = np.array([np.inf, np.inf], dtype=np.float64)

    # Objective vector for "maximize c^T x"
    c = np.array([-3.0, -5.0], dtype=np.float64)

    # Build a Model object directly from arrays (no disk I/O)
    model = Model.from_arrays(A, AL, AU, l, u, c)

    # ------------------------------------------------------------------
    # 2. Set parameters
    # ------------------------------------------------------------------
    param = Parameters()
    param.time_limit = 3600.0
    param.stoptol = 1e-9
    param.device_number = 0  # GPU id; set to 0 by default

    # ------------------------------------------------------------------
    # 3. Solve & print result
    # ------------------------------------------------------------------
    res = model.solve(param)

    print("status:     ", getattr(res, "output_type", None))
    print("objective:  ", getattr(res, "primal_obj", None))
    print("solution x: ", getattr(res, "x", None))
    print("iterations: ", getattr(res, "iter", None))
    print("solve_time: ", getattr(res, "time", None))


if __name__ == "__main__":
    main()
