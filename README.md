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
Run roam benchmarks as:
```
python benchmark_roam.py
```

The results for the each model can be seen in "./results/model_name/[ms|ss]/bs_[batch_size]".

<!-- Run heuristics benchmarks as:
```
python benchmark_heu.py
```

Run MODeL benchmarks as:
```
bash benchmark_no_acc.sh
``` -->

# Reference
At last, we would like to express our gratitude to MODeL [<sup>1</sup>](#MODeL) for providing significant inspiration and guidance in the construction of the DNN graph for our project.

<div id="MODeL"></div>
- [1] [MODeL](https://github.com/facebookresearch/MODel_opt)