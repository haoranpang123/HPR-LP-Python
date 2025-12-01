import os
import sys
import argparse

# --- Ensure the project root is on sys.path so we can import the local `src` package ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.types import HPRLP_parameters
from src.utils import run_dataset


def _str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Run HPR-LP over a dataset directory of .mps files."
    )
    # ---- paths ----
    p.add_argument(
        "--data_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "example"),
        help="Path to a directory containing .mps files.",
    )
    p.add_argument(
        "--result_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "result_example"),
        help="Directory to write CSV/log outputs.",
    )

    # ---- solver tolerances / limits ----
    p.add_argument(
        "--time_limit", 
        type=float, 
        default=3600.0, 
        help="Time limit (sec)."
    )
    p.add_argument(
        "--stoptol", 
        type=float, 
        default=1e-8, 
        help="Stopping tolerance."
    )
    p.add_argument(
        "--max_iter",
        type=int,
        default=2**31 - 1,
        help="Maximum number of iterations.",
    )
    p.add_argument(
        "--check_iter",
        type=int,
        default=150,
        help="Interval for checking termination.",
    )

    # ---- device / warm-up ----
    p.add_argument(
        "--device_number", 
        type=int, 
        default=0, 
        help="CUDA device number."
    )
    p.add_argument(
        "--warm_up", 
        type=_str2bool, 
        default=True, 
        help="Do warm-up run once (true/false)."
    )
    p.add_argument(
        "--print_frequency",
        type=int,
        default=-1,
        help="Log print frequency (-1 for auto).",
    )

    # ---- scaling options ----
    p.add_argument(
        "--use_Ruiz_scaling",
        type=_str2bool,
        default=True,
        help="Enable Ruiz scaling.",
    )
    p.add_argument(
        "--use_Pock_Chambolle_scaling",
        type=_str2bool,
        default=True,
        help="Enable Pock-Chambolle scaling.",
    )
    p.add_argument(
        "--use_bc_scaling",
        type=_str2bool,
        default=True,
        help="Enable (b,c) scaling.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data_path = os.path.abspath(os.path.expanduser(args.data_path))
    result_path = os.path.abspath(os.path.expanduser(args.result_path))

    # Initialize parameters
    params = HPRLP_parameters()
    
    # Map arguments to params
    params.time_limit = float(args.time_limit)
    params.stoptol = float(args.stoptol)
    params.max_iter = int(args.max_iter)
    params.check_iter = int(args.check_iter)
    
    params.device_number = int(args.device_number)
    params.warm_up = bool(args.warm_up)
    params.print_frequency = int(args.print_frequency)

    params.use_Ruiz_scaling = bool(args.use_Ruiz_scaling)
    params.use_Pock_Chambolle_scaling = bool(args.use_Pock_Chambolle_scaling)
    params.use_bc_scaling = bool(args.use_bc_scaling)

    # Run the batch
    run_dataset(data_path, result_path, params)


if __name__ == "__main__":
    main()