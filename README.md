# HPR-LP: A GPU Solver for Linear Programming

> **The Python implementation of the Halpern Peaceman-Rachford (HPR) method for solving linear programming (LP) problems on GPUs**

---

## LP Problem Formulation

<div align="center">

$$
\begin{array}{ll}
\underset{x \in \mathbb{R}^n}{\min} \quad & \langle c, x \rangle \\
\text{s.t.} \quad & L \leq A x \leq U, \\
& l \leq x \leq u .
\end{array}
$$

</div>

## Getting Started

### 1. Prerequisites
Before using HPR-LP, make sure the following dependencies are installed:
- **CUDA** (Install the appropriate version for your GPU)

Build a new environment with conda and activate it.

```bash
conda create -n hprlp_python python=3.12
conda activate hprlp_python
```

### 2. Install PyTorch First

Install `torch` and `torchvision` that match your CUDA / GPU setup.  
Example (CUDA 12.8 wheels):

```bash
pip3 install torch torchvision
```

If that doesn’t work, please install PyTorch by following the official install instructions for your system. PyTorch is not in requirements.txt because the right build depends on your machine.

---

### 3. Install the Remaining Dependencies

After PyTorch is installed, install the rest of the required packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Running Examples

The demo LP problem is:

### Test Problem (LP)

The default demo problem we ship is a small linear program of the form


$$
\begin{aligned}
\min \quad & 3 x_1 + 5 x_2\\
\text{s.t.} \quad & x_1 + 2 x_2 \le 10, \\
& 3 x_1 + x_2 \le 12, \\
& x_1 \ge 0, \\
& x_2 \ge 0.
\end{aligned}
$$





After installing the environment, you can run the solver in two ways: **model-based API** or **`.mps` API**.

### 1. Model-based API 

You can also build and solve a small demo script under `demo/demo_model_api.py`.

```bash
python demo/demo_model_api.py
```


### 2. MPS File API

```bash
python scripts/run_single_file.py
```

This calls `run_single_file.py` with its default arguments, which solves one LP model in MPS format.

You can also override any of these from the command line, for example:

```bash
python scripts/run_single_file.py \
    --file_name /path/to/your_problem.mps \
    --device_number 0 \
    --stoptol 1e-8 \
    --time_limit 3600 \
    --check_iter 150 \
    --warm_up True \
    --use_Ruiz_scaling True \
    --use_Pock_Chambolle_scaling True \
    --use_bc_scaling True
```

## Parameters
Solver configuration:
- `max_iter` - Maximum iterations (default: unlimited)
- `stop_tol` - Stopping tolerance (default: 1e-4)
- `time_limit` - Time limit in seconds (default: 3600)
- `device_number` - CUDA device ID (default: 0)
- `check_iter` - Convergence check interval (default: 150)
- `use_Ruiz_scaling` - Ruiz scaling (default: True)
- `use_Pock_Chambolle_scaling` - Pock-Chambolle scaling (default: True)
- `use_bc_scaling` - Bounds/cost scaling (default: True)

## Results
Solution information:
- `status` - "OPTIMAL", "TIME_LIMIT", "MAX_ITER", "ERROR"
- `x` - Primal solution
- `y` - Dual solution
- `primal_obj` - Primal objective value
- `gap` - Duality gap
- `residuals` - Final KKT residual
- `iter` - Total iterations
- `time` - Solve time (seconds)
- `iter4/6/8` - Iterations to reach 1e-4/6/8 tolerance
- `time4/6/8` - Time to reach tolerance

## Available HPR-LP Implementations

For the C implementation on GPUs, please visit the following repository:

[https://github.com/PolyU-IOR/HPR-LP-C](https://github.com/PolyU-IOR/HPR-LP-C)

For the Julia implementation on CPUs and GPUs, please visit the following repository:

[https://github.com/PolyU-IOR/HPR-LP](https://github.com/PolyU-IOR/HPR-LP)

For the Python implementation on GPUs, please visit the following repository:

[https://github.com/PolyU-IOR/HPR-LP-Python](https://github.com/PolyU-IOR/HPR-LP-Python)

For the MATLAB implementation on CPUs, please visit the following repository:  

[https://github.com/PolyU-IOR/HPR-LP-MATLAB](https://github.com/PolyU-IOR/HPR-LP-MATLAB)


## Contributors

### Core Team
- **Kaihuang Chen** - Contributor
- **Jiawei Lu** - Developer
- **Defeng Sun** - Principal Investigator
- **Yancheng Yuan** - Contributor
- **Guojun Zhang** - Contributor
- **Xinyuan Zhao** - Contributor

### Acknowledgments
- Community contributors and testers



## Reference

If you use this project in your research, please put a star on this repository and cite:
```bibtex
@article{chen2025hpr,
  title={HPR-LP: An implementation of an HPR method for solving linear programming},
  author={Chen, Kaihuang and Sun, Defeng and Yuan, Yancheng and Zhang, Guojun and Zhao, Xinyuan},
  journal={Mathematical Programming Computation},
  pages={1--28},
  year={2025},
  publisher={Springer}
}
```

## License

This project is licensed under the MIT License.

Copyright (c) 2025 HPR-LP develop team

See the [LICENSE](LICENSE) file for full details.
