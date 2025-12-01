# src/utils.py
from __future__ import annotations
import os
import time
import warnings
from typing import Tuple, Union
import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix, csr_matrix
from contextlib import redirect_stdout

from .types import LPStub, LP_info_gpu, Scaling_info_gpu

warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------------------------------------------------------
# 1) Formulation
# -----------------------------------------------------------------------------
def formulation_gpu(lp: LPStub, device: str) -> LP_info_gpu:
    dev = torch.device(device)
    m0, n = int(lp.ncon), int(lp.nvar)

    rows = np.asarray(lp.arows, dtype=np.int64)
    cols = np.asarray(lp.acols, dtype=np.int64)
    vals = np.asarray(lp.avals, dtype=np.float64)
    A_coo = coo_matrix((vals, (rows, cols)), shape=(m0, n))
    A_coo.sum_duplicates()
    A_csr = A_coo.tocsr()

    AL = np.asarray(lp.lcon, dtype=np.float64)
    AU = np.asarray(lp.ucon, dtype=np.float64)
    c  = np.asarray(lp.c,    dtype=np.float64)
    l  = np.asarray(lp.lvar, dtype=np.float64)
    u  = np.asarray(lp.uvar, dtype=np.float64)
    c0 = float(lp.c0)

    abs_rowsum = np.asarray(np.abs(A_csr).sum(axis=1)).ravel()
    drop = (abs_rowsum == 0.0) | (np.isneginf(AL) & np.isposinf(AU))
    if np.any(drop):
        keep = np.where(~drop)[0]
        A_csr = A_csr[keep, :]
        AL, AU = AL[keep], AU[keep]
        print(f"Deleted {int(drop.sum())} rows.")

    AT_csr = A_csr.transpose().tocsr()

    def _to_torch(csr, d):
        crow = torch.as_tensor(csr.indptr, dtype=torch.int64, device=d)
        ccol = torch.as_tensor(csr.indices, dtype=torch.int64, device=d)
        cval = torch.as_tensor(csr.data, dtype=torch.float64, device=d)
        return torch.sparse_csr_tensor(crow, ccol, cval, size=csr.shape, dtype=torch.float64, device=d)

    A_gpu = _to_torch(A_csr, dev)
    AT_gpu = _to_torch(AT_csr, dev)
    
    to64 = lambda x: torch.as_tensor(x, dtype=torch.float64, device=dev)
    return LP_info_gpu(A_gpu, AT_gpu, to64(c), to64(AL), to64(AU), to64(l), to64(u), c0)


def lpstub_from_arrays(
    A,
    AL,
    AU,
    l,
    u,
    c,
    c0: float = 0.0,
) -> LPStub:
    """
    Build an LPStub directly from NumPy / SciPy arrays.

    Parameters
    ----------
    A   : np.ndarray or scipy.sparse matrix
        Constraint matrix of shape (m, n).
    AL  : array-like of shape (m,)
        Lower bounds on A x.
    AU  : array-like of shape (m,)
        Upper bounds on A x.
    l   : array-like of shape (n,)
        Lower bounds on variables.
    u   : array-like of shape (n,)
        Upper bounds on variables.
    c   : array-like of shape (n,)
        Objective vector for maximize c^T x.
    c0  : float, optional
        Constant term in the objective (default 0.0).
    """

    # 1) Make A into a CSR float64 matrix
    if not isinstance(A, csr_matrix):
        A = csr_matrix(A, dtype=np.float64)
    else:
        A = A.astype(np.float64)

    m, n = A.shape

    # 2) Make bounds and objective into float64 arrays
    AL = np.asarray(AL, dtype=np.float64)
    AU = np.asarray(AU, dtype=np.float64)
    l  = np.asarray(l,  dtype=np.float64)
    u  = np.asarray(u,  dtype=np.float64)
    c  = np.asarray(c,  dtype=np.float64)
    c0 = float(c0)

    # 3) Basic shape checks (raise a clear error if mismatch)
    if AL.shape[0] != m or AU.shape[0] != m:
        raise ValueError(f"Row bounds AL/AU must have length {m}, "
                         f"got {AL.shape[0]} and {AU.shape[0]}.")
    if l.shape[0] != n or u.shape[0] != n or c.shape[0] != n:
        raise ValueError(f"Variable bounds and c must have length {n}, "
                         f"got l={l.shape[0]}, u={u.shape[0]}, c={c.shape[0]}.")

    # 4) Convert A to COO to extract row/col/value arrays
    A_coo = A.tocoo()
    A_coo.sum_duplicates()

    arows = A_coo.row.astype(np.int64)
    acols = A_coo.col.astype(np.int64)
    avals = A_coo.data.astype(np.float64)

    # 5) Build LPStub
    lp = LPStub(
        arows=arows,
        acols=acols,
        avals=avals,
        ncon=m,
        nvar=n,
        lcon=AL,
        ucon=AU,
        lvar=l,
        uvar=u,
        c=c,
        c0=c0,
    )
    return lp

# -----------------------------------------------------------------------------
# 2) Scaling
# -----------------------------------------------------------------------------
def scaling_gpu(lp: LP_info_gpu, use_Ruiz, use_PC, use_bc) -> Tuple[LP_info_gpu, Scaling_info_gpu]:
    device = lp.A.device
    dtype = torch.float64
    m, n = lp.A.shape
    l_org, u_org = lp.l.clone(), lp.u.clone()

    def _norm_b(AL, AU):
        ALn, AUn = AL.clone(), AU.clone()
        ALn[torch.isneginf(ALn)] = 0; AUn[torch.isposinf(AUn)] = 0
        return torch.linalg.norm(torch.maximum(ALn.abs(), AUn.abs())).item()

    norm_b_org = 1.0 + _norm_b(lp.AL, lp.AU)
    norm_c_org = 1.0 + torch.linalg.norm(lp.c).item()

    A_coo = lp.A.to_sparse_coo().coalesce()
    row_norm = torch.ones(m, dtype=dtype, device=device)
    col_norm = torch.ones(n, dtype=dtype, device=device)

    def _reduce(idx_dim, reduce_op):
        idx = A_coo.indices()[idx_dim]
        val = A_coo.values().abs()
        size = m if idx_dim == 0 else n
        out = torch.zeros(size, dtype=dtype, device=device)
        if reduce_op == "max": out.scatter_reduce_(0, idx, val, reduce="amax", include_self=True)
        else: out.scatter_add_(0, idx, val)
        out[out == 0] = 1.0
        return out

    def _apply_scale(r, c):
        idx = A_coo.indices()
        vals = A_coo.values()
        rr = r.index_select(0, idx[0])
        cc = c.index_select(0, idx[1])
        vals.div_(rr * cc)
        lp.AL.div_(r); lp.AU.div_(r)
        lp.c.div_(c); lp.l.mul_(c); lp.u.mul_(c)
        row_norm.mul_(r); col_norm.mul_(c)

    if use_Ruiz:
        for _ in range(10):
            r = torch.sqrt(_reduce(0, "max"))
            c = torch.sqrt(_reduce(1, "max"))
            _apply_scale(r, c)

    if use_PC:
        r = torch.sqrt(_reduce(0, "sum"))
        c = torch.sqrt(_reduce(1, "sum"))
        _apply_scale(r, c)

    b_scale = c_scale = 1.0
    if use_bc:
        b_scale = 1.0 + _norm_b(lp.AL, lp.AU)
        c_scale = 1.0 + torch.linalg.norm(lp.c).item()
        lp.AL.div_(b_scale); lp.AU.div_(b_scale); lp.l.div_(b_scale); lp.u.div_(b_scale)
        lp.c.div_(c_scale)

    lp.A = A_coo.to_sparse_csr()
    lp.AT = lp.A.transpose(0, 1).to_sparse_csr()
    
    norm_b = _norm_b(lp.AL, lp.AU)
    norm_c = torch.linalg.norm(lp.c).item()
    
    sc = Scaling_info_gpu(
        l_org, u_org, row_norm, col_norm, 
        float(b_scale), float(c_scale), 
        float(norm_b), float(norm_c), 
        float(norm_b_org), float(norm_c_org)
    )
    return lp, sc

# -----------------------------------------------------------------------------
# 3) Power Iteration
# -----------------------------------------------------------------------------
def power_iteration_gpu(A, AT, max_iter=5000, tol=1e-4):
    m, n = A.shape
    g = torch.Generator(device=A.device)
    g.manual_seed(1)
    z = torch.randn(m, dtype=torch.float64, device=A.device, generator=g) + 1e-8
    q = torch.zeros(m, dtype=torch.float64, device=A.device)
    
    # Allocate ATq buffer outside the loop to avoid frequent memory allocation
    ATq = torch.zeros(n, dtype=torch.float64, device=A.device)
    
    lam = 1.0
    for _ in range(max_iter):
        q.copy_(z)
        q.div_(torch.linalg.vector_norm(q))
        
        # ATq = AT * q (using addmm)
        # We reuse the pre-allocated ATq buffer. beta=0.0 ensures we overwrite previous values.
        ATq.view(-1,1).addmm_(AT, q.view(-1,1), beta=0.0, alpha=1.0)
        
        # z = A * ATq
        z.zero_()
        z.view(-1,1).addmm_(A, ATq.view(-1,1), beta=0.0, alpha=1.0)
        
        lam = torch.dot(q, z).item()
        
        # check err: q = z - lam*q
        q.mul_(-lam).add_(z)
        if torch.linalg.vector_norm(q).item() < tol:
            return lam
            
    return lam

# -----------------------------------------------------------------------------
# 4) Runners (Original Logging Format Restored)
# -----------------------------------------------------------------------------
def run_lpstub(lp_raw: LPStub, params):
    from .algorithm import HPRLP

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Install a CUDA-enabled PyTorch.")

    device = torch.device(f"cuda:{params.device_number}")
    torch.cuda.set_device(device)

    # (1) Formulation
    t_form0 = time.perf_counter()
    lp_gpu = formulation_gpu(lp_raw, device)
    form_time = time.perf_counter() - t_form0
    print(f"FORMULATING LP time: {form_time:.2f} seconds")

    # (2) Scaling
    t_scale0 = time.perf_counter()
    lp_gpu, sc_gpu = scaling_gpu(
        lp_gpu,
        use_Ruiz=params.use_Ruiz_scaling,
        use_PC=params.use_Pock_Chambolle_scaling,
        use_bc=params.use_bc_scaling,
    )
    scale_time = time.perf_counter() - t_scale0
    print(f"SCALING LP time: {scale_time:.2f} seconds")

    # (3) Solve
    print("main run starts: ----------------------------------------------------------------------------------------------------------")
    t_solve0 = time.perf_counter()
    results = HPRLP(lp_gpu, sc_gpu, params)
    solve_time = getattr(results, "time", time.perf_counter() - t_solve0)
    print("main run ends----------------------------------------------------------------------------------------------------------")

    setup_time = form_time + scale_time
    total_time = setup_time + solve_time
    print(
        f"Total time: {total_time:.2f}s"
        f"  setup time = {setup_time:.2f}s"
        f"  solve time = {solve_time:.2f}s"
    )
    return results


def run_file(file_name: str, params):
    from .io_mps import read_lp_to_lpstub
    from .algorithm import HPRLP

    if not file_name.endswith(".mps"):
        raise ValueError("Please provide a .mps file")
    if not os.path.isfile(file_name):
        raise FileNotFoundError(f"MPS file not found: {file_name}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Install a CUDA-enabled PyTorch.")

    device = torch.device(f"cuda:{params.device_number}")
    torch.cuda.set_device(device)

    # 1) Read
    print(f"READING FILE ... {file_name}")
    t_read0 = time.perf_counter()
    lp_raw = read_lp_to_lpstub(file_name)
    read_time = time.perf_counter() - t_read0
    print(f"READING FILE time: {read_time:.2f} seconds")

    # 2) Formulation
    print("FORMULATING LP ...")
    t_form0 = time.perf_counter()
    lp_gpu = formulation_gpu(lp_raw, device)
    form_time = time.perf_counter() - t_form0
    print(f"FORMULATING LP time: {form_time:.2f} seconds")

    # 3) Scaling
    print("SCALING LP ...")
    t_scale0 = time.perf_counter()
    lp_gpu, sc_gpu = scaling_gpu(
        lp_gpu,
        use_Ruiz=params.use_Ruiz_scaling,
        use_PC=params.use_Pock_Chambolle_scaling,
        use_bc=params.use_bc_scaling,
    )
    scale_time = time.perf_counter() - t_scale0
    print(f"SCALING LP time: {scale_time:.2f} seconds")

    # 4) Solve
    print("main run starts: ----------------------------------------------------------------------------------------------------------")
    t_solve0 = time.perf_counter()
    results = HPRLP(lp_gpu, sc_gpu, params)
    solve_time = getattr(results, "time", time.perf_counter() - t_solve0)
    print("main run ends----------------------------------------------------------------------------------------------------------")

    # 5) Summary
    setup_time = form_time + scale_time
    total_time = read_time + setup_time + solve_time
    print(
        f"Total time: {total_time:.2f}s"
        f"  read time = {read_time:.2f}s"
        f"  setup time = {setup_time:.2f}s"
        f"  solve time = {solve_time:.2f}s"
    )
    return results

def run_dataset(data_path, result_path, params):
    import sys
    import numpy as np
    import pandas as pd
    from contextlib import redirect_stdout

    print("HPR-LP version v0.1.3")
    files = sorted(os.listdir(data_path))

    os.makedirs(result_path, exist_ok=True)
    csv_file = os.path.join(result_path, "HPRLP_result.csv")
    log_path = os.path.join(result_path, "HPRLP_log.txt")

    if os.path.isfile(csv_file):
        df_prev = pd.read_csv(csv_file)
        df_prev = df_prev[~df_prev["name"].isin(["SGM10", "solved"])].copy()
        namelist = df_prev["name"].tolist()
        iterlist = df_prev["iter"].tolist()
        timelist = df_prev["alg_time"].tolist()
        reslist = df_prev["res"].tolist()
        objlist = df_prev["primal_obj"].tolist()
        statuslist = df_prev["status"].tolist()
        iter4list = df_prev["iter_4"].tolist()
        time4list = df_prev["time_4"].tolist()
        iter6list = df_prev["iter_6"].tolist()
        time6list = df_prev["time_6"].tolist()
        iter8list = df_prev["iter_8"].tolist()
        time8list = df_prev["time_8"].tolist()
    else:
        namelist, iterlist, timelist, reslist, objlist, statuslist = [], [], [], [], [], []
        iter4list, time4list, iter6list, time6list, iter8list, time8list = [], [], [], [], [], []

    warm_up_done = False
    with open(log_path, "a", encoding="utf-8") as log_f:
        for i, file in enumerate(files, 1):
            if (file in namelist) or (not file.endswith(".mps")):
                continue

            FILE_NAME = os.path.join(data_path, file)
            print(f"solving the problem {i}: {file}")

            with redirect_stdout(log_f):
                print(f"solving the problem {i}: {file}")
                if getattr(params, "warm_up", False) and not warm_up_done:
                    warm_up_done = True
                    print("warm up starts: ---------------------------------------------------------------------------------------------------------- ")
                    t0 = time.perf_counter()
                    orig_max_iter = params.max_iter
                    try:
                        params.max_iter = 200
                        _ = run_file(FILE_NAME, params)
                    finally:
                        params.max_iter = orig_max_iter
                    print(f"warm up time: {time.perf_counter() - t0}")
                    print("warm up ends ----------------------------------------------------------------------------------------------------------")

                print("main run starts: ----------------------------------------------------------------------------------------------------------")
                t_all0 = time.perf_counter()
                results = run_file(FILE_NAME, params)
                all_time = time.perf_counter() - t_all0
                print("main run ends----------------------------------------------------------------------------------------------------------")

                print(
                    "iter = ", results.iter,
                    f"  time = {results.time:3.2e}",
                    f"  residual = {results.residuals:3.2e}",
                    f"  primal_obj = {results.primal_obj:3.15e}",
                )

                namelist.append(file)
                iterlist.append(results.iter)
                timelist.append(min(results.time, params.time_limit))
                reslist.append(results.residuals)
                objlist.append(results.primal_obj)
                statuslist.append(results.output_type)
                iter4list.append(results.iter_4)
                time4list.append(min(results.time_4, params.time_limit))
                iter6list.append(results.iter_6)
                time6list.append(min(results.time_6, params.time_limit))
                iter8list.append(results.iter_8)
                time8list.append(min(results.time_8, params.time_limit))

            df = pd.DataFrame({
                "name": namelist, "iter": iterlist, "alg_time": timelist,
                "res": reslist, "primal_obj": objlist, "status": statuslist,
                "iter_4": iter4list, "time_4": time4list,
                "iter_6": iter6list, "time_6": time6list,
                "iter_8": iter8list, "time_8": time8list,
            })
            
            shift = 10.0
            def sgm(x):
                arr = np.asarray(x, dtype=float)
                return float(np.exp(np.mean(np.log(arr + shift))) - shift) if len(arr) > 0 else 0.0

            df_sgm = pd.DataFrame([{
                "name": "SGM10",
                "iter": sgm(iterlist), "alg_time": sgm(timelist),
                "res": "", "primal_obj": "", "status": "",
                "iter_4": sgm(iter4list), "time_4": sgm(time4list),
                "iter_6": sgm(iter6list), "time_6": sgm(time6list),
                "iter_8": sgm(iter8list), "time_8": sgm(time8list),
            }])

            solved = int(np.sum(np.asarray(timelist) < params.time_limit))
            solved_4 = int(np.sum(np.asarray(time4list) < params.time_limit))
            solved_6 = int(np.sum(np.asarray(time6list) < params.time_limit))
            solved_8 = int(np.sum(np.asarray(time8list) < params.time_limit))

            df_solved = pd.DataFrame([{
                "name": "solved", "iter": "", "alg_time": solved, "res": "", "primal_obj": "", "status": "",
                "iter_4": "", "time_4": solved_4, "iter_6": "", "time_6": solved_6, "iter_8": "", "time_8": solved_8,
            }])

            df_out = pd.concat([df, df_sgm, df_solved], ignore_index=True)
            df_out.to_csv(csv_file, index=False, encoding="utf-8")

    print(f"The solver has finished running the dataset, total {len(files)} problems")

def run_single(file_name: str, params):
    print("HPR-LP python version v0.1.0")
    print(f"data path: {file_name}")

    if not file_name.endswith(".mps"):
        raise ValueError("The file is not in the correct format, please provide a .mps file")

    if getattr(params, "warm_up", False):
        print("warm up starts: ---------------------------------------------------------------------------------------------------------- ")
        t0 = time.perf_counter()
        orig_max_iter = params.max_iter
        try:
            params.max_iter = 200
            _ = run_file(file_name, params)
        finally:
            params.max_iter = orig_max_iter
        print(f"warm up time: {time.perf_counter() - t0}")
        print("warm up ends ----------------------------------------------------------------------------------------------------------")

    print("main run starts: ----------------------------------------------------------------------------------------------------------")
    results = run_file(file_name, params)
    print("main run ends----------------------------------------------------------------------------------------------------------")
    return results