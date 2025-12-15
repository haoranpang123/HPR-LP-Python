# src/io_mps.py
"""
io_mps.py — Minimal reader for .mps / .lp using HiGHS (highspy) only.
Optimized for HPR-LP: outputs data types strictly compatible with LPStub.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from scipy.sparse import coo_matrix

# --- highspy (HiGHS) ---
try:
    from highspy import Highs, HighsStatus
except ImportError as e:
    raise ImportError(
        "highspy is required. Install with:\n"
        "  conda install -c conda-forge highspy\n"
        "or:\n"
        "  pip install highspy"
    ) from e

# --- Project Imports ---
from .types import LPStub

# --- Helpers ---
def _as_infty(x, thr: float = 1e199) -> np.ndarray:
    """Map very large magnitudes (~1e200) to ±np.inf."""
    arr = np.asarray(x, dtype=np.float64).copy()
    arr[arr <= -thr] = -np.inf
    arr[arr >= +thr] = +np.inf
    return arr

def _to_coo_from_csc(n_row: int, n_col: int,
                     starts: List[int], index: List[int], value: List[float]
                     ) -> Tuple[List[int], List[int], List[float]]:
    """Convert HiGHS CSC (column-wise) to COO triplets."""
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    
    # Pre-allocate if possible could be faster, but lists are fine for read-once.
    for j in range(n_col):
        p0, p1 = starts[j], starts[j + 1]
        if p0 == p1: continue
        
        # Extend is slightly faster than append loop
        r_slice = index[p0:p1]
        v_slice = value[p0:p1]
        
        rows.extend(r_slice)
        cols.extend([j] * (p1 - p0))
        vals.extend(v_slice)
        
    return rows, cols, vals

# --- Public API ---
def read_lp_to_lpstub(filename: str, *, relax_integer: bool = True) -> LPStub:
    """
    Read an LP/MIP from .mps/.lp using HiGHS and convert to LPStub.
    Strictly returns an src.types.LPStub object.
    """
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {filename}")

    # 1. Read model using Highs
    h = Highs()
    status = h.readModel(str(path))
    
    ok = (status == 0)
    try:
        ok = ok or (status == HighsStatus.kOk) or (status == HighsStatus.kWarning)
    except Exception:
        pass
        
    if not ok:
        raise RuntimeError(f"HiGHS failed to read model: {filename} (status={status})")

    # 2. Extract LP structure
    lp = h.getLp()
    nvar = int(getattr(lp, "num_col_", 0))
    ncon = int(getattr(lp, "num_row_", 0))

    # 3. Objective (costs) and constant offset
    c  = np.array(getattr(lp, "col_cost_", []), dtype=np.float64)
    c0 = float(getattr(lp, "offset_", 0.0))

    # Handle Objective Sense (Minimization standard)
    flip = False
    try:
        sense = h.getObjectiveSense()
        # HighsObjectiveSense.kMaximize usually maps to a value (often 2 or -1)
        s_str = str(sense).lower()
        if "max" in s_str: 
            flip = True
    except Exception:
        # Fallback to internal property if enum fails
        val = int(getattr(lp, "sense_", 1))
        if val == 2: # 1=min, 2=max often
            flip = True

    if flip:
        c = -c
        c0 = -c0

    # 4. Bounds
    lvar = _as_infty(getattr(lp, "col_lower_", []))
    uvar = _as_infty(getattr(lp, "col_upper_", []))
    lcon = _as_infty(getattr(lp, "row_lower_", []))
    ucon = _as_infty(getattr(lp, "row_upper_", []))

    # 5. Matrix (CSC -> COO triplets)
    a_mat = getattr(lp, "a_matrix_", None)
    
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []

    if a_mat is not None:
        starts = list(getattr(a_mat, "start_", []))
        index  = list(getattr(a_mat, "index_", []))
        value  = list(getattr(a_mat, "value_", []))
        
        if len(starts) == nvar + 1:
            rows, cols, vals = _to_coo_from_csc(ncon, nvar, starts, index, value)

    # 6. Integrality Check (Optional Warning)
    if not relax_integer:
        try:
            model = h.getModel()
            col_type = getattr(model, "col_type_", None)
            if col_type is not None:
                ct = np.asarray(col_type, dtype=int)
                if (ct != 0).any(): # 0=continuous
                    raise ValueError("Integer variables found but relax_integer=False.")
        except AttributeError:
            pass

    # 7. Construct and Return LPStub
    # We explicitly convert lists to numpy arrays here to match LPStub expected types
    # though LPStub.__post_init__ also handles conversion.
    
    return LPStub(
        ncon=ncon,
        nvar=nvar,
        arows=np.array(rows, dtype=np.int64),
        acols=np.array(cols, dtype=np.int64),
        avals=np.array(vals, dtype=np.float64),
        lcon=lcon,
        ucon=ucon,
        lvar=lvar,
        uvar=uvar,
        c=c,
        c0=c0
    )
