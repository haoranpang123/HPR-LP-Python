# src/kernels.py
from __future__ import annotations
import torch

# -----------------------------------------------------------------------------
# 1) X/Z update (PyTorch In-Place Optimized)
# -----------------------------------------------------------------------------
def update_x_z_gpu_compiled(
    ws_ATy, ws_x, ws_z_bar, ws_x_bar, ws_x_hat, ws_dx, ws_last_x, 
    ws_AT, ws_y, ws_c, ws_l, ws_u, 
    sigma, fact1, fact2, to_check
):
    """
    X/Z update using PyTorch in-place operations.
    SpMV: ws_ATy = AT * y (using addmm_ with beta=0 for overwrite).
    """
    
    # SpMV: ATy = AT * y
    # Using addmm_ on a view to perform Sparse x Dense -> Dense (Zero Allocation)
    ws_ATy.view(-1, 1).addmm_(ws_AT, ws_y.view(-1, 1), beta=0.0, alpha=1.0)

    if to_check:
        # z_bar = ATy - c
        torch.sub(ws_ATy, ws_c, out=ws_z_bar) 
        
        # z_bar = x + sigma * z_bar
        ws_z_bar.mul_(sigma).add_(ws_x)

        # x_bar = clamp(z_bar, l, u)
        ws_x_bar.copy_(ws_z_bar)
        torch.maximum(ws_x_bar, ws_l, out=ws_x_bar)
        torch.minimum(ws_x_bar, ws_u, out=ws_x_bar)

        # z_bar = (x_bar - z_bar_old) / sigma
        ws_z_bar.sub_(ws_x_bar).mul_(-1.0).div_(sigma)

        # x_hat = 2 * x_bar - x
        torch.mul(ws_x_bar, 2.0, out=ws_x_hat)
        ws_x_hat.sub_(ws_x)

        # dx = x_bar - x_hat
        torch.sub(ws_x_bar, ws_x_hat, out=ws_dx)

        # x = fact1 * last_x + fact2 * x_hat
        torch.mul(ws_last_x, fact1, out=ws_x)
        ws_x.add_(ws_x_hat, alpha=fact2)

    else:
        # Reuse ws_x_hat as temporary buffer for z_temp
        torch.sub(ws_ATy, ws_c, out=ws_x_hat)
        ws_x_hat.mul_(sigma).add_(ws_x)

        # x_clamped -> store in ws_dx (temp)
        torch.maximum(ws_x_hat, ws_l, out=ws_dx)
        torch.minimum(ws_dx, ws_u, out=ws_dx)
        
        # x_hat = 2 * x_clamped - x
        ws_x_hat.copy_(ws_dx).mul_(2.0).sub_(ws_x)
        
        # x update
        torch.mul(ws_last_x, fact1, out=ws_x)
        ws_x.add_(ws_x_hat, alpha=fact2)


# -----------------------------------------------------------------------------
# 2) Y update (PyTorch In-Place Optimized)
# -----------------------------------------------------------------------------
def update_y_gpu_compiled(
    ws_Ax, ws_y, ws_y_bar, ws_y_obj, ws_dy, ws_y_hat, ws_last_y, 
    ws_A, ws_x_hat, ws_AL, ws_AU,
    lambda_max, sigma, Halpern_fact1, Halpern_fact2, to_check
):
    # SpMV: Ax = A * x_hat (Zero Allocation)
    ws_Ax.view(-1, 1).addmm_(ws_A, ws_x_hat.view(-1, 1), beta=0.0, alpha=1.0)
    
    fact1 = lambda_max * sigma
    fact2 = 1.0 / fact1

    if to_check:
        # v = Ax - fact1 * y
        torch.mul(ws_y, -fact1, out=ws_y_obj)
        ws_y_obj.add_(ws_Ax) # y_obj now holds 'v'
        
        # d = max(AL - v, min(AU - v, 0))
        # Use ws_dy as temp buffer for 'd'
        torch.sub(ws_AU, ws_y_obj, out=ws_dy)
        ws_dy.clamp_max_(0.0)
        
        torch.sub(ws_AL, ws_y_obj, out=ws_y_bar)
        
        torch.maximum(ws_y_bar, ws_dy, out=ws_y_bar) # y_bar holds 'd'
        
        # y_obj = v + d
        ws_y_obj.add_(ws_y_bar)
        
        # yb = d * fact2
        ws_y_bar.mul_(fact2)
        
        # yh = 2 * yb - y
        torch.mul(ws_y_bar, 2.0, out=ws_y_hat)
        ws_y_hat.sub_(ws_y)
        
        # dy = yb - yh
        torch.sub(ws_y_bar, ws_y_hat, out=ws_dy)
        
        # y update
        torch.mul(ws_last_y, Halpern_fact1, out=ws_y)
        ws_y.add_(ws_y_hat, alpha=Halpern_fact2)
        
    else:
        # Optimized path
        torch.mul(ws_y, -fact1, out=ws_dy)
        ws_dy.add_(ws_Ax) # dy holds 'v'
        
        # d calculation using y_hat as temp
        torch.sub(ws_AU, ws_dy, out=ws_y_hat)
        ws_y_hat.clamp_max_(0.0)
        
        # reuse ws_y_obj as temp buffer
        torch.sub(ws_AL, ws_dy, out=ws_y_obj)
        
        # d = max(..., ...)
        torch.maximum(ws_y_obj, ws_y_hat, out=ws_y_hat)
        
        # yb = d * fact2
        ws_y_hat.mul_(fact2)
        
        # yh = 2 * yb - y
        ws_y_hat.mul_(2.0).sub_(ws_y)
        
        # y update
        torch.mul(ws_last_y, Halpern_fact1, out=ws_y)
        ws_y.add_(ws_y_hat, alpha=Halpern_fact2)


# -----------------------------------------------------------------------------
# 3) Residual Kernels
# -----------------------------------------------------------------------------
def axpby_gpu(a: float, x: torch.Tensor, b: float, y: torch.Tensor, out: torch.Tensor) -> None:
    out.copy_(x).mul_(a).add_(y, alpha=b)

def compute_Rd_gpu(ws, sc) -> None:
    # ATy = AT * y_bar
    ws.ATy.view(-1, 1).addmm_(ws.AT, ws.y_bar.view(-1, 1), beta=0.0, alpha=1.0)
    
    # Rd = (c - ATy - z_bar) * col_norm
    ws.Rd.copy_(ws.c).sub_(ws.ATy).sub_(ws.z_bar).mul_(sc.col_norm)

def compute_err_Rp_gpu(ws, sc) -> None:
    # Ax = A * x_bar
    ws.Ax.view(-1, 1).addmm_(ws.A, ws.x_bar.view(-1, 1), beta=0.0, alpha=1.0)
    
    # Rp = min(AU - Ax, 0)
    torch.sub(ws.AU, ws.Ax, out=ws.Rp)
    ws.Rp.clamp_max_(0.0)
    
    # tmp = AL - Ax
    tmp = ws.AL - ws.Ax
    
    # Rp = max(Rp, tmp)
    torch.maximum(ws.Rp, tmp, out=ws.Rp)
    
    # Scale
    ws.Rp.mul_(sc.row_norm)

def compute_err_lu_kernel(col_norm, dx, x, l, u, n):
    torch.sub(l, x, out=dx)
    dx.clamp_min_(0.0)
    tmp = x - u
    tmp.clamp_min_(0.0)
    dx.add_(tmp).div_(col_norm)