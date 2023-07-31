import os
import sys
import math
import time
import json
import yaml
import torch
import argparse
import torchvision
import torchtext
import pandas as pd
import multiprocessing
from collections import defaultdict
from multiprocessing import Process, Pool, Manager
from dill import pickle, dumps, loads

from roam.olla import torch_graph_importer
from roam.tools import utils, visualizer
from roam.tools.load_models import load_model
from roam.decomposer import subtask_info
from roam.decomposer.subtask_generator import subtask_generator
from roam.decomposer.key_points import compute_spans, find_key_points, get_all_activations
from roam.lao_and_reducer.subtask_solver import Scheduler
from roam.lao_scheduler.sublao_init import generate_schedulers
from roam.lao_scheduler.sublao_run import run_scheduler


def err_call_back(err):
    print("Error: {}".format(str(err)))


if __name__ == "__main__":
    sys.setrecursionlimit(100000)

    BENCHMARKS = {
        "alexnet": {
            "modes": ["ms", "ss"],
            "bs": [1, 32],
        },
        "vgg": { 
            "modes": ["ms", "ss"],
            "bs": [1, 32],
        },
        "mnasnet": {
            "modes": ["ms", "ss"],
            "bs": [1, 32],
        },
        "mobilenet": {
            "modes": ["ms", "ss"],
            "bs": [1, 32],
        },
        "efficientnet": {
            "modes": ["ms", "ss"],
            "bs": [1, 32],
        },
        "vit": {
            "modes": ["ms", "ss"],
            "bs": [1, 32],
        },
        "xlmr": {
            "modes": ["ms", "ss"],
            "bs": [1, 32],
        },
        "bert": {
            "modes": ["ms", "ss"],
            "bs": [1, 32],
        },
        "gpt2-XL": {
            "modes": ["ss"],
            "bs": [1, 2, 4],
        }
    }
 
    parser = argparse.ArgumentParser(description="benchmark roam")
    parser.add_argument("-m", "--model", "--models", nargs="+", type=str, default=BENCHMARKS.keys()) 
    parser.add_argument("-n", "--num_iters", type=int, default=1)
    parser.add_argument("--batch_size", nargs="+", type=int)
    parser.add_argument("--modes", nargs="+", type=str)
    parser.add_argument("--dump-orders", action="store_true")

    # Default number of iterations.
    f = open("./roam/decomposer/config.yaml")
    settings = yaml.load(f, Loader=yaml.Loader)
    args = parser.parse_args()
    num_iters = args.num_iters
    batch_size = args.batch_size
    modes = args.modes

    orders = defaultdict(lambda: {})
    model_names = []
    bss = []
    num_ops = []
    avg_times = defaultdict(lambda: [])
    for model_name in args.model:
        if not batch_size:
            batch_size = BENCHMARKS[model_name]["bs"]

        for bs in batch_size:
            if bs not in BENCHMARKS[model_name]["bs"]:
                print("Skip benchmark for {} at batch size {}.".format(model_name, bs))
                continue

            print("Running {}/{}...".format(model_name, bs))

            model_names.append(model_name)
            bss.append(bs)

            ratio = settings[model_name]["r"+str(bs)]
            if ratio == None:
                ratio = settings["default"]["r"+str(bs)]

            if not modes:
                modes = BENCHMARKS[model_name]["modes"]

            for mode in modes:
                if mode not in BENCHMARKS[model_name]["modes"]:
                    print("Skip benchmark for {} in {}.".format(model_name, mode))
                    continue

                NL = settings[model_name][mode]["NL"]
                TL = settings[model_name][mode]["TL"]
                if not NL:
                    NL = settings["default"][mode]["NL"]
                if not TL:
                    TL = settings["default"][mode]["TL"]
                
                graph, graph_name, size, num_nodes, pt_node_order = load_model(model_name, bs, opti=True)
                global_scheduler, schedulers, all_activations = generate_schedulers(model_name, graph, mode, int(NL), int(TL), int(ratio))
                p_num = len(schedulers)
                
                path = [f"./logs/graphs/{model_name}/{mode}", f"./results/{model_name}/{mode}"]
                for p in path:
                    if not os.path.exists(p):
                        os.makedirs(p)

                times = []
                required_memorys = []
                simulated_memorys = []
                fragmentations = []
                for iter in range(num_iters):
                    try:
                        start = time.time()

                        # Global vars.
                        schedules_manager = Manager()
                        schedules = schedules_manager.dict()

                        pool = Pool(p_num)
                        for id in range(p_num):
                            scheduler = schedulers[id]
                            input_scheduler = dumps(scheduler)
                            
                            stage = pool.apply_async(
                                run_scheduler, 
                                args=(input_scheduler, id, model_name, graph_name, mode, schedules,),
                                error_callback=err_call_back    
                            )

                        pool.close()
                        pool.join()

                        # Validate whether schedules result is complete.
                        # and change dict into list.
                        schedules_list = [0] * len(schedulers)
                        flag = True
                        for i in range(len(schedulers)):
                            if i not in schedules:
                                flag = False
                                print("Subtask {} failed.".format(i))
                            elif flag:
                                # Reconstruct schedules with edges in prime graph.
                                old_schedule = loads(schedules[i])
                                new_schedule = {}
                                for e, s in old_schedule.items():
                                    ee = graph.edges[e.name]
                                    new_schedule[ee] = s
                                schedules_list[0-i-1] = new_schedule

                        if not flag:
                            sys.exit(-1)

                        tensor_schedule_map = {}
                        for i in range(len(schedules_list)):
                            if schedules_list[i] == 0:
                                sys.exit(-1)
                            
                            # current_schedule = loads(schedules_list[i])
                            for e, _ in schedules_list[i].items():
                                if e not in tensor_schedule_map:
                                    tensor_schedule_map[e] = [i]
                                else:
                                    tensor_schedule_map[e].append(i)

                        overall_schedule, required_memory, mem_locations = global_scheduler.reduce(
                            schedules_list, 
                            all_activations, 
                            tensor_schedule_map
                        )
                        end = time.time()
                        required_memory = size + required_memory / (1024**2)
                        print("Solved {}/{}/{} [{}]/[{}]: {}MiB in {}s.".format(model_name, mode, bs, iter+1, num_iters, required_memory, end - start))
                        
                        # print("Overall schedule: {}".format(overall_schedule))

                        utils.validate_tensors(graph, overall_schedule)
                        utils.validate_timeline(overall_schedule)
                        state, node_order = utils.validate_node_ordering(graph, overall_schedule)
                        utils.validate_address_allocation(mem_locations)

                        node_order = sorted(node_order.items(), key=lambda x:x[1])
                        simulated_memory, _ = utils.run_simulation(graph, node_order)
                        simulated_memory = size + simulated_memory / (1024**2)
                        simulated_memorys.append(simulated_memory)

                        times.append(end - start)
                        required_memorys.append(required_memory)
                        fragmentation = (required_memory - simulated_memory) / required_memory
                        fragmentations.append(fragmentation)
                        
                        visualizer.draw_schedule(overall_schedule, img_path=f"./logs/graphs/{model_name}/{mode}/" + graph_name + "_overall" + ".png")
                    
                    except Exception as e:
                        print("FAILED TO OPTIMIZE {}: {}.".format(model_name, e))

                avg_time = sum(times) / num_iters
                avg_required_memory = sum(required_memorys) / num_iters
                avg_simulated_memory = sum(simulated_memorys) / num_iters
                avg_fragmentation = sum(fragmentations) / num_iters
                
                avg_times[mode].append(avg_time)

                times.append("avg: {:.2f}".format(avg_time))
                required_memorys.append("avg: {:.2f}MiB".format(avg_required_memory))
                simulated_memorys.append("avg: {:.2f}MiB".format(avg_simulated_memory))
                fragmentations.append("avg: {:.2f}%".format(avg_fragmentation * 100))
                csv_name = "./results/{}/{}/bs_{}.csv".format(model_name, mode, bs)
                df = pd.DataFrame({
                                    "time": times,
                                    "required memory": required_memorys, 
                                    "simulated memory": simulated_memorys, 
                                    "fragmentation": fragmentations}
                                )
                df.to_csv(csv_name, index=False)
                print("Optimize {}/{}/{}: {}MiB in {}s.".format(model_name, mode, bs, avg_required_memory, avg_time))
            num_ops.append(num_nodes)

    df = pd.DataFrame({
        "model name": model_names,
        "batch size": bss,
        "number of operators": num_ops,
        "ms time": avg_times["ms"],
        "ss time": avg_times["ss"]
    })
    csv_name = "./results/op_time.csv"
    df.to_csv(csv_name, index=False)