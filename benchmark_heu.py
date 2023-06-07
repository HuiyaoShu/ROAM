import sys
import json
import time
import torch
import argparse
import torchtext
import torchvision

import pandas as pd
from collections import defaultdict
from intervaltree import IntervalTree, Interval
from rota.tools import utils, visualizer
from rota.tools.load_models import load_model
from rota.olla import torch_graph_importer
from baselines.heuristics import LESCEA, LLFB

BENCHMARKS = {
    "alexnet": [1, 32],
    "vgg": [1, 32],
    "vit": [1, 32],
    "xlmr": [1, 32],
    "bert": [1, 32],
    "mnasnet": [1, 32],
    "mobilenet": [1, 32],
    "efficientnet": [1, 32],
    "gpt2-XL": [1, 2, 4],
}

modes = [
    "e2e"
]

default_iter = 1
model_names = []
simulated_memorys = defaultdict(lambda: [])
e2e_memorys = defaultdict(lambda: [])
e2e_times = defaultdict(lambda: [])
fragmentations = defaultdict(lambda: [])

parser = argparse.ArgumentParser(description="benchmark solver")
parser.add_argument("-m", "--model", nargs="+", type=str, default=BENCHMARKS.keys()) 
parser.add_argument("--batch_size", nargs="+", type=int)
args = parser.parse_args()
models = args.model

for model in models:
    if model not in BENCHMARKS.keys():
        print("Skip model {}.".format(model))
        continue
    model_names.append(model)

    if args.batch_size:
        batch_size = args.batch_size
    else:
        batch_size = BENCHMARKS[model]
    
    for bs in batch_size:
        if bs not in BENCHMARKS[model]:
            print("Skip batch size {} for model {}.".format(bs, model))

        print("Heuristic Optimization for {}-bs{}.".format(model, bs))
        graph, graph_name, size, num_nodes, pt_node_order = load_model(model, bs, opti=False)
        tensor_size = [e.size for e in graph.edges.values()]
        gcd = utils._GCD(tensor_size)
        
        for mode in modes:
            if mode == "e2e":
                max_simulated_memory = 0
                max_required_memory = 0
                max_fragmentation = -1
                ntime = 0
                for i in range(default_iter):
                    start = time.time()
                    order, memory = LESCEA.list_memory_scheduler(graph)
                    simulated_memory = size + memory / (1024**2)

                    memory, mem_loc = LLFB.long_lived_first(graph, order, gcd)
                    required_memory = size + memory / (1024**2)
                    end = time.time()
                    ntime += end - start
                    
                    frag = (required_memory - simulated_memory) / required_memory
                    if frag > max_fragmentation:
                        max_fragmentation = frag
                        max_simulated_memory = simulated_memory
                        max_required_memory = required_memory

                    state = utils.validate_address_allocation(mem_loc)
                    if state:
                        print("[pass] [{}]/[{}] {}/{} address ok.".format(i, default_iter, graph_name, mode))
                
                ntime /= default_iter
                simulated_memorys[bs].append(max_simulated_memory)
                e2e_memorys[bs].append(max_required_memory)
                e2e_times[bs].append(ntime)

                # Just temporary.
                # frag = (required_memory - simulated_memory) / required_memory
                fragmentations[bs].append("{:.2f}%".format(max_fragmentation * 100))

                print("- Simulated memory: {}.".format(max_simulated_memory))
                print("- E2e Required memory: {}/{}s.".format(max_required_memory, ntime))
                print("- E2e fragmentation: {}.".format("{:.2f}%".format(max_fragmentation * 100)))

            elif mode == "frag":
                order = orders[model]
                simulated_memory = utils.run_simulation(graph, order)
                
                required_memory = 0
                for i in range(default_iter):
                    memory, mem_loc = long_lived_first(graph, order, gcd)
                    required_memory = max(required_memory, size + memory / (1024**2))
                    state = utils.validate_address_allocation(mem_loc)
                    if state:
                        print("[pass] [{}]/[{}] {}/{} address ok.".format(i, default_iter, graph_name, mode))
                
                fragmentation = (required_memory - simulated_memory) / required_memory
                fragmentations[bs].append("{:.2f}%".format(fragmentation * 100))

                print("- Required memory with optimized order: {}.".format(required_memory))
                print("- Fragmentation: {}.".format("{:.2f}%".format(fragmentation * 100)))

df_1 = pd.DataFrame({
    "model": model_names,
    "simulated memmory": simulated_memorys[1],
    "e2e memory": e2e_memorys[1],
    "fragmentation": fragmentations[1],
    "e2e time": e2e_times[1]
})

csv_1 = "./results/baseline_heu_1.csv"
df_1.to_csv(csv_1, index=False)

df_32 = pd.DataFrame({
    "model": model_names,
    "simulated memory": simulated_memorys[32],
    "e2e memory": e2e_memorys[32],
    "fragmentation": fragmentations[32],
    "e2e time": e2e_times[32]
})
csv_32 = "./results/baseline_heu_32.csv"
df_32.to_csv(csv_32, index=False)            