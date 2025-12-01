# src/algorithm.py
import time
import math
import torch
from typing import Sequence, Optional

from .types import (
    HPRLP_parameters, HPRLP_results,
    HPRLP_workspace_gpu, HPRLP_residuals, HPRLP_restart,
    LP_info_gpu, Scaling_info_gpu,
)
from .kernels import (
    update_x_z_gpu_compiled,
    update_y_gpu_compiled,
    compute_Rd_gpu,
    compute_err_Rp_gpu,
    compute_err_lu_kernel,
    axpby_gpu,
)
from .utils import power_iteration_gpu

# -----------------------------------------------------------------------------
# Helper Functions (Defined BEFORE HPRLP to avoid NameError)
# -----------------------------------------------------------------------------

def allocate_workspace_gpu(lp: LP_info_gpu, sc: Scaling_info_gpu) -> HPRLP_workspace_gpu:
    device = lp.c.device
    m, n = lp.A.shape

    ws = HPRLP_workspace_gpu()
    ws.m, ws.n = int(m), int(n)

    # Allocate tensors
    ws.x      = torch.zeros(n, dtype=torch.float64, device=device)
    ws.x_hat  = torch.zeros_like(ws.x)
    ws.x_bar  = torch.zeros_like(ws.x)
    ws.dx     = torch.zeros_like(ws.x)

    ws.y      = torch.zeros(m, dtype=torch.float64, device=device)
    ws.y_hat  = torch.zeros_like(ws.y)
    ws.y_bar  = torch.zeros_like(ws.y)
    ws.dy     = torch.zeros_like(ws.y)
    ws.y_obj  = torch.zeros_like(ws.y)

    ws.z_bar  = torch.zeros_like(ws.x)

    # References
    ws.A  = lp.A
    ws.AT = lp.AT
    ws.AL = lp.AL
    ws.AU = lp.AU
    ws.c  = lp.c
    ws.l  = lp.l
    ws.u  = lp.u

    # Residual buffers
    ws.Rp  = torch.zeros(m, dtype=torch.float64, device=device)
    ws.Rd  = torch.zeros(n, dtype=torch.float64, device=device)
    
    # SpMV buffers (Dense)
    ws.Ax  = torch.zeros(m, dtype=torch.float64, device=device)
    ws.ATy = torch.zeros(n, dtype=torch.float64, device=device)

    ws.last_x = torch.zeros(n, dtype=torch.float64, device=device)
    ws.last_y = torch.zeros(m, dtype=torch.float64, device=device)

    ws.to_check   = False
    ws.lambda_max = 0.0

    if sc.norm_b > 1e-8 and sc.norm_c > 1e-8:
        ws.sigma = sc.norm_b / sc.norm_c
    else:
        ws.sigma = 1.0

    return ws


def initialize_restart(sigma: float) -> HPRLP_restart:
    r = HPRLP_restart()
    r.first_restart = True
    r.save_gap = float("inf")
    r.current_gap = float("inf")
    r.last_gap = float("inf")
    r.best_gap = float("inf")
    r.best_sigma = float(sigma)
    return r


def print_step(it: int) -> int:
    if it <= 0: return 10
    return max(int(10 ** (math.floor(math.log10(it))) / 10), 10)


def compute_maximum_eigenvalue(lp, ws, params) -> float:
    t0 = time.time()
    print("ESTIMATING MAXIMUM EIGENVALUE ...")
    if params.use_gpu:
        torch.cuda.synchronize()
    lam = power_iteration_gpu(lp.A, lp.AT) * 1.01
    if params.use_gpu:
        torch.cuda.synchronize()
    power_time = time.time() - t0
    ws.lambda_max = lam
    print(f"ESTIMATING MAXIMUM EIGENVALUE time = {power_time:.2f} seconds")
    print(f"estimated maximum eigenvalue of AAT = {lam:.2e}")
    return power_time


@torch.no_grad()
def compute_weighted_norm_gpu(ws) -> float:
    # SpMV: Ax = A * dx (In-Place using addmm_ for speed)
    ws.Ax.zero_()
    ws.Ax.view(-1, 1).addmm_(ws.A, ws.dx.view(-1, 1), beta=0.0, alpha=1.0)

    dot_prod = 2.0 * torch.dot(ws.Ax, ws.dy)
    dy_sq    = torch.dot(ws.dy, ws.dy)
    dx_sq    = torch.dot(ws.dx, ws.dx)

    weighted = ws.sigma * (ws.lambda_max * dy_sq) + dx_sq / ws.sigma + dot_prod

    if weighted < 0:
        print(f"The estimated maximum eigenvalue is too small! Current value is {ws.lambda_max}")
        # Only fetch scalar to CPU when absolutely necessary
        val_fix = -(dot_prod + dx_sq / ws.sigma) / (ws.sigma * dy_sq) * 1.05
        ws.lambda_max = float(val_fix.item())
        print(f"The new estimated maximum eigenvalue is {ws.lambda_max}")
        
        fallback = (-(dot_prod + dx_sq / ws.sigma) * 0.05)
        return float(torch.sqrt(fallback).item())

    return float(torch.sqrt(weighted).item())


@torch.no_grad()
def compute_residuals_gpu(
    ws: HPRLP_workspace_gpu,
    lp: LP_info_gpu,
    sc: Scaling_info_gpu,
    res: HPRLP_residuals,
    iter: int,
) -> None:
    scbc = sc.b_scale * sc.c_scale
    obj_const = lp.obj_constant
    
    # Dot products
    p_obj = torch.dot(ws.c, ws.x_bar).item()
    d_obj_y = torch.dot(ws.y_obj, ws.y_bar).item()
    d_obj_z = torch.dot(ws.x_bar, ws.z_bar).item()
    
    res.primal_obj_bar = scbc * p_obj + obj_const
    res.dual_obj_bar   = scbc * (d_obj_y + d_obj_z) + obj_const
    
    denom = 1.0 + abs(res.primal_obj_bar) + abs(res.dual_obj_bar)
    res.rel_gap_bar = abs(res.primal_obj_bar - res.dual_obj_bar) / denom

    # Dual Residuals
    compute_Rd_gpu(ws, sc)
    res.err_Rd_org_bar = sc.c_scale * torch.norm(ws.Rd).item() / sc.norm_c_org

    # Primal Residuals
    compute_err_Rp_gpu(ws, sc)
    res.err_Rp_org_bar = sc.b_scale * torch.norm(ws.Rp).item() / sc.norm_b_org

    # Box violation (only at iter 0 usually)
    if iter == 0:
        compute_err_lu_kernel(sc.col_norm, ws.dx, ws.x_bar, ws.l, ws.u, int(ws.n))
        res.err_Rp_org_bar = max(res.err_Rp_org_bar, sc.b_scale * torch.norm(ws.dx).item())

    res.KKTx_and_gap_org_bar = max(res.err_Rp_org_bar, res.err_Rd_org_bar, res.rel_gap_bar)


@torch.no_grad()
def update_sigma_gpu(
    restart: HPRLP_restart,
    ws: HPRLP_workspace_gpu,
    res: HPRLP_residuals,
) -> None:
    if 1 <= restart.restart_flag <= 3:
        axpby_gpu(1.0, ws.x_bar, -1.0, ws.last_x, ws.dx)
        axpby_gpu(1.0, ws.y_bar, -1.0, ws.last_y, ws.dy)

        primal_move = torch.norm(ws.dx).item()
        dual_move   = torch.norm(ws.dy).item()

        if 1e-16 < primal_move < 1e12 and 1e-16 < dual_move < 1e12:
            pm_over_dm = primal_move / dual_move
            sqrtl = math.sqrt(ws.lambda_max)
            ratio = pm_over_dm / sqrtl

            fact = math.exp(-0.05 * (restart.current_gap / restart.best_gap))
            temp_1 = max(
                min(res.err_Rd_org_bar, res.err_Rp_org_bar),
                min(res.rel_gap_bar, restart.current_gap),
            )
            sigma_cand = math.exp(fact * math.log(ratio) + (1.0 - fact) * math.log(restart.best_sigma))

            if temp_1 > 9e-10:
                kappa = 1.0
            elif temp_1 > 5e-10:
                ratio_infeas_org = res.err_Rd_org_bar / res.err_Rp_org_bar
                kappa = max(min(math.sqrt(ratio_infeas_org), 100.0), 1e-2)
            else:
                ratio_infeas_org = res.err_Rd_org_bar / res.err_Rp_org_bar
                kappa = max(min(ratio_infeas_org, 100.0), 1e-2)

            ws.sigma = kappa * sigma_cand
        else:
            ws.sigma = 1.0


def check_restart(restart: HPRLP_restart, it: int, check_iter: int, sigma: float) -> None:
    restart.restart_flag = 0

    if restart.first_restart:
        if it == check_iter:
            restart.first_restart = False
            restart.restart_flag = 1
            restart.best_gap = restart.current_gap
            restart.best_sigma = sigma               
    else:
        if it % check_iter == 0:
            if restart.current_gap < 0.0:
                restart.current_gap = 1e-6
                print("current_gap < 0")            

            if restart.current_gap <= 0.2 * restart.last_gap:
                restart.sufficient += 1
                restart.restart_flag = 1

            if (restart.current_gap <= 0.6 * restart.last_gap) and \
               (restart.current_gap > 1.00 * restart.save_gap):
                restart.necessary += 1
                restart.restart_flag = 2

            if restart.inner >= 0.2 * it:
                restart.long += 1
                restart.restart_flag = 3

            if restart.best_gap > restart.current_gap:
                restart.best_gap = restart.current_gap
                restart.best_sigma = sigma           

            restart.save_gap = restart.current_gap


@torch.no_grad()
def do_restart(restart: HPRLP_restart, ws: HPRLP_workspace_gpu) -> None:
    if restart.restart_flag > 0:
        ws.x.copy_(ws.x_bar)
        ws.y.copy_(ws.y_bar)
        ws.last_x.copy_(ws.x_bar)
        ws.last_y.copy_(ws.y_bar)

        restart.times += 1
        restart.inner = 0
        restart.save_gap = float("inf")


def check_break(res: HPRLP_residuals, it: int, t_start: float, params: HPRLP_parameters) -> str:
    if res.KKTx_and_gap_org_bar < params.stoptol:
        return "OPTIMAL"
    if it >= params.max_iter:
        return "MAX_ITER"
    if time.time() - t_start > params.time_limit:
        return "TIME_LIMIT"
    return "CONTINUE"


@torch.no_grad()
def collect_results_gpu(
    ws: HPRLP_workspace_gpu,
    residuals: HPRLP_residuals,
    sc: Scaling_info_gpu,
    iter: int,
    t_start_alg: float,
    power_time: float,
    status: Optional[str] = None,
    tolerance_times: Optional[Sequence[float]] = None,
    tolerance_iters: Optional[Sequence[int]] = None,
) -> HPRLP_results:
    if status is None: status = "UNKNOWN"
    if tolerance_times is None: tolerance_times = (0.0, 0.0, 0.0)
    if tolerance_iters is None: tolerance_iters = (0, 0, 0)

    tt = list(tolerance_times) + [0.0]*3
    ti = list(tolerance_iters) + [0]*3

    # Rescale to original space (CPU)
    x_scaled = (ws.x_bar * sc.b_scale) / sc.col_norm
    y_scaled = (ws.y_bar * sc.c_scale) / sc.row_norm
    z_scaled = (ws.z_bar * sc.c_scale) * sc.col_norm

    out = HPRLP_results()
    out.x = x_scaled.cpu().numpy()
    out.y = y_scaled.cpu().numpy()
    out.z = z_scaled.cpu().numpy()

    out.iter = int(iter)
    out.time = float(time.time() - t_start_alg)
    out.power_time = float(power_time)

    def to_float(v):
        return float(v.item()) if isinstance(v, torch.Tensor) else float(v)

    out.residuals  = to_float(residuals.KKTx_and_gap_org_bar)
    out.primal_obj = to_float(residuals.primal_obj_bar)
    out.gap        = to_float(residuals.rel_gap_bar)
    out.output_type = status

    out.time_4 = out.time if (tt[0] == 0.0) else float(tt[0])
    out.iter_4 = out.iter if (ti[0] == 0)   else int(ti[0])
    out.time_6 = out.time if (tt[1] == 0.0) else float(tt[1])
    out.iter_6 = out.iter if (ti[1] == 0)   else int(ti[1])
    out.time_8 = out.time if (tt[2] == 0.0) else float(tt[2])
    out.iter_8 = out.iter if (ti[2] == 0)   else int(ti[2])

    return out


# -----------------------------------------------------------------------------
# 2) Main Function HPRLP
# -----------------------------------------------------------------------------

def HPRLP(lp, sc, params: HPRLP_parameters) -> HPRLP_results:
    """
    HPR-LP main driver (GPU-only), aligned with Julia `solve`.
    """
    t_start_alg = time.time()

    # --- Initialization ---
    ws: HPRLP_workspace_gpu = allocate_workspace_gpu(lp, sc)
    restart: HPRLP_restart = initialize_restart(ws.sigma)
    res: HPRLP_residuals = HPRLP_residuals()

    # --- Compile Kernels ---
    print("Compiling Kernels with torch.compile (mode='reduce-overhead')...")
    try:
        # We compile the functions imported from kernels.py
        ws.opt_update_x_z = torch.compile(update_x_z_gpu_compiled, mode="reduce-overhead")
        ws.opt_update_y   = torch.compile(update_y_gpu_compiled,   mode="reduce-overhead")
    except Exception as e:
        print(f"Warning: torch.compile failed ({e}), falling back to eager execution.")
        ws.opt_update_x_z = update_x_z_gpu_compiled
        ws.opt_update_y   = update_y_gpu_compiled

    # --- Power iteration ---
    power_time = compute_maximum_eigenvalue(lp, ws, params)

    # --- Log header ---
    print(" iter      errRp        errRd         p_obj           d_obj           gap         sigma       time")

    tolerance_levels = [1e-4, 1e-6, 1e-8]
    tolerance_times = [0.0, 0.0, 0.0]
    tolerance_iters = [0, 0, 0]
    tolerance_reached = [False, False, False]

    # --- Main loop ---
    for iter in range(0, params.max_iter + 1):
        if params.print_frequency == -1:
            print_yes = ((iter % print_step(iter) == 0) or
                         (iter == params.max_iter) or
                         (time.time() - t_start_alg > params.time_limit))
        elif params.print_frequency > 0:
            print_yes = ((iter % params.print_frequency == 0) or
                         (iter == params.max_iter) or
                         (time.time() - t_start_alg > params.time_limit))
        else:
            raise ValueError(f"Invalid print_frequency: {params.print_frequency}")

        # --- residuals ---
        if (iter % params.check_iter == 0) or print_yes:
            compute_residuals_gpu(ws, lp, sc, res, iter)

        # --- stopping & restart checks ---
        status = check_break(res, iter, t_start_alg, params)
        check_restart(restart, iter, params.check_iter, ws.sigma)

        # --- logging ---
        if print_yes or (status != "CONTINUE"):
            elapsed = time.time() - t_start_alg
            print(f"{iter:5d}    "
                  f"{res.err_Rp_org_bar:3.2e}    "
                  f"{res.err_Rd_org_bar:3.2e}    "
                  f"{res.primal_obj_bar:+7.6e}    "
                  f"{res.dual_obj_bar:+7.6e}    "
                  f"{res.rel_gap_bar:3.2e}    "
                  f"{ws.sigma:3.2e}    "
                  f"{elapsed:6.2f}")

        # --- tolerance milestones ---
        for i, tol in enumerate(tolerance_levels):
            if (not tolerance_reached[i]) and (res.KKTx_and_gap_org_bar < tol):
                tolerance_times[i] = time.time() - t_start_alg
                tolerance_iters[i] = iter
                tolerance_reached[i] = True
                print(f"KKT < {tol} at iter = {iter}")

        # --- termination ---
        if status != "CONTINUE":
            print(f"Termination reason: {status}, accuracy = {res.KKTx_and_gap_org_bar}")
            out = collect_results_gpu(ws, res, sc, iter, t_start_alg, power_time, status, tolerance_times, tolerance_iters)
            return out

        # --- update sigma ---
        update_sigma_gpu(restart, ws, res)

        # --- restart if needed ---
        do_restart(restart, ws)

        # --- schedule bar points ---
        ws.to_check = ((iter + 1) % params.check_iter == 0) or (restart.restart_flag > 0)
        if params.print_frequency == -1:
            ws.to_check = ws.to_check or ((iter + 1) % print_step(iter + 1) == 0)
        elif params.print_frequency > 0:
            ws.to_check = ws.to_check or ((iter + 1) % params.print_frequency == 0)

        # --- one HPR step (Compiled Kernels) ---
        fact1 = 1.0 / (restart.inner + 2.0)
        fact2 = 1.0 - fact1
        
        # Call X/Z (Pass buffer ws.ATy)
        ws.opt_update_x_z(
            ws.ATy, # Output Buffer
            ws.x, ws.z_bar, ws.x_bar, ws.x_hat, ws.dx, ws.last_x,
            ws.AT, ws.y, ws.c, ws.l, ws.u,
            ws.sigma, fact1, fact2, ws.to_check
        )

        # Call Y (Pass buffer ws.Ax)
        ws.opt_update_y(
            ws.Ax, # Output Buffer
            ws.y, ws.y_bar, ws.y_obj, ws.dy, ws.y_hat, ws.last_y,
            ws.A, ws.x_hat, ws.AL, ws.AU,
            ws.lambda_max, ws.sigma, fact1, fact2, ws.to_check
        )
        
        restart.inner += 1

        # --- weighted norm ---
        if (iter + 1) % params.check_iter == 0:
            restart.current_gap = compute_weighted_norm_gpu(ws)
        if restart.restart_flag > 0:
            restart.last_gap = compute_weighted_norm_gpu(ws)

    # fallthrough
    out = collect_results_gpu(ws, res, sc, params.max_iter, t_start_alg, power_time, "MAX_ITER", tolerance_times, tolerance_iters)
    return out