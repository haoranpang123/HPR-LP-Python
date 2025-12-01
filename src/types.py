# src/types.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Sequence, Union, Any
import numpy as np
import torch

@dataclass
class HPRLP_parameters:
    stoptol: float = 1e-6
    max_iter: int = 2**31 - 1
    time_limit: float = 3600.0
    check_iter: int = 150
    use_Ruiz_scaling: bool = True
    use_Pock_Chambolle_scaling: bool = True
    use_bc_scaling: bool = True
    use_gpu: bool = True
    device_number: int = 0
    warm_up: bool = True
    print_frequency: int = -1

@dataclass
class HPRLP_results:
    iter: int = 0
    iter_4: int = 0
    iter_6: int = 0
    iter_8: int = 0
    time: float = math.nan
    time_4: float = math.nan
    time_6: float = math.nan
    time_8: float = math.nan
    power_time: float = math.nan
    primal_obj: float = math.nan
    residuals: float = math.nan
    gap: float = math.nan
    output_type: str = ""
    x: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    z: Optional[np.ndarray] = None

@dataclass
class HPRLP_workspace_gpu:
    """Workspace using PyTorch Tensors (float64 on CUDA)."""
    x: Optional[torch.Tensor] = None
    x_hat: Optional[torch.Tensor] = None
    x_bar: Optional[torch.Tensor] = None
    dx: Optional[torch.Tensor] = None

    y: Optional[torch.Tensor] = None
    y_hat: Optional[torch.Tensor] = None
    y_bar: Optional[torch.Tensor] = None
    y_obj: Optional[torch.Tensor] = None
    dy: Optional[torch.Tensor] = None

    z_bar: Optional[torch.Tensor] = None

    A: Optional[torch.Tensor] = None
    AT: Optional[torch.Tensor] = None
    AL: Optional[torch.Tensor] = None
    AU: Optional[torch.Tensor] = None
    c: Optional[torch.Tensor] = None
    l: Optional[torch.Tensor] = None
    u: Optional[torch.Tensor] = None

    Rp: Optional[torch.Tensor] = None
    Rd: Optional[torch.Tensor] = None
    
    # Buffers for SpMV results (In-Place Optimized)
    Ax: Optional[torch.Tensor] = None
    ATy: Optional[torch.Tensor] = None
    
    last_x: Optional[torch.Tensor] = None
    last_y: Optional[torch.Tensor] = None

    m: int = 0
    n: int = 0
    sigma: float = 1.0
    lambda_max: float = 0.0
    to_check: bool = False
    
    # Compiled kernel handles
    opt_update_x_z: Any = None
    opt_update_y: Any = None

@dataclass
class HPRLP_residuals:
    err_Rp_org_bar: float = math.nan
    err_Rd_org_bar: float = math.nan
    primal_obj_bar: float = math.nan
    dual_obj_bar: float = math.nan
    rel_gap_bar: float = math.nan
    KKTx_and_gap_org_bar: float = math.nan

@dataclass
class HPRLP_restart:
    restart_flag: int = 0
    first_restart: bool = True
    last_gap: float = 0.0
    current_gap: float = 0.0
    save_gap: float = 0.0
    best_gap: float = float("inf")
    best_sigma: float = 1.0
    inner: int = 0
    step: int = 0
    sufficient: int = 0
    necessary: int = 0
    long: int = 0
    ratio: float = 0.0
    times: int = 0
    weighted_norm: float = 0.0

@dataclass
class LP_info_gpu:
    A: torch.Tensor
    AT: torch.Tensor
    c: torch.Tensor
    AL: torch.Tensor
    AU: torch.Tensor
    l: torch.Tensor
    u: torch.Tensor
    obj_constant: float

    def __post_init__(self):
        if self.A.layout != torch.sparse_csr:
             raise TypeError("A must be torch.sparse_csr layout.")
        if self.AT.layout != torch.sparse_csr:
             raise TypeError("AT must be torch.sparse_csr layout.")

@dataclass
class Scaling_info_gpu:
    l_org: torch.Tensor
    u_org: torch.Tensor
    row_norm: torch.Tensor
    col_norm: torch.Tensor
    b_scale: float
    c_scale: float
    norm_b: float
    norm_c: float
    norm_b_org: float
    norm_c_org: float

IntSeq = Union[Sequence[int], np.ndarray]
FloatSeq = Union[Sequence[float], np.ndarray]

@dataclass
class LPStub:
    arows: IntSeq
    acols: IntSeq
    avals: FloatSeq
    ncon: int
    nvar: int
    lcon: np.ndarray
    ucon: np.ndarray
    lvar: np.ndarray
    uvar: np.ndarray
    c: np.ndarray
    c0: float = 0.0

    def __post_init__(self):
        self.arows = np.asarray(self.arows, dtype=np.int64)
        self.acols = np.asarray(self.acols, dtype=np.int64)
        self.avals = np.asarray(self.avals, dtype=np.float64)
        self.lcon = np.asarray(self.lcon, dtype=np.float64)
        self.ucon = np.asarray(self.ucon, dtype=np.float64)
        self.lvar = np.asarray(self.lvar, dtype=np.float64)
        self.uvar = np.asarray(self.uvar, dtype=np.float64)
        self.c    = np.asarray(self.c,    dtype=np.float64)