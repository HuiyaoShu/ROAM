# ROTA
ROTA (Reorder Operators and Arrange Tenors Address to Reduce Memory Usage of Deep Learning) enables effective and efficient tensor allocations optimization.

# Setup
Prepare python environment:
```
pip install .
```

Prepare Gurobi:
```
export LD_LIBRARY_PATH="${LIB_PATH}"
export GRB_LICENSE_FILE="${LICENSE_PATH}"
```

Note: ROTA is tested with Gurobi v9.1.1. Choose the version as needed and prepare the license. 

# Benchmarks
Run benchmarks as:
```
python benchmark_solver.py
```

The results for the benchmarked model can be seen in "./results/model_name/[ms|ss]/bs_[batch_size]".
