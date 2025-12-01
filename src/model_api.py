# src/model_api.py
from __future__ import annotations
import numpy as np
from scipy import sparse
from typing import Union

from .types import (
    LPStub,
    HPRLP_parameters,
    HPRLP_results,
)
from .utils import (
    lpstub_from_arrays,
    run_lpstub,
)

class Parameters(HPRLP_parameters):
    """
    Thin wrapper so user code can:
        param = src.Parameters()
        param.stoptol = 1e-9
        param.device_number = 0
    Internally this IS your HPRLP_parameters.
    """
    pass


class Model:
    """
    High-level LP container.

    Internally we just store an LPStub (the same structure you already pass
    to formulation_gpu in utils.py).
    """

    def __init__(self, lpstub: LPStub):
        self._lpstub = lpstub

    @staticmethod
    def from_arrays(
        A: Union[np.ndarray, sparse.spmatrix],
        AL: np.ndarray,
        AU: np.ndarray,
        l:  np.ndarray,
        u:  np.ndarray,
        c:  np.ndarray,
        c0: float = 0.0,
    ) -> "Model":
        """
        Build a Model from in-memory arrays, no .mps needed.
        Solver convention: maximize c^T x.
        """
        lpstub = lpstub_from_arrays(A, AL, AU, l, u, c, c0=c0)
        return Model(lpstub)

    def solve(self, param: Parameters) -> HPRLP_results:
        """
        Call the main GPU solver (formulation_gpu -> scaling_gpu -> HPRLP).
        """
        return run_lpstub(self._lpstub, param)
