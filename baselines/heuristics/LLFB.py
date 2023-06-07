import sys
import json
import torch
import torchtext
import torchvision
from rota.tools import utils, visualizer
from collections import defaultdict
from intervaltree import IntervalTree, Interval


def extract_spans(graph, order):
    execute_time = {}
    for node, time in order.items():
        execute_time[node] = time
    
    MUL = {}
    for e in graph.edges.values():
        if e.source not in order:
            continue
        
        lb = execute_time[e.source]
        ub = -1
        for sink in e.sinks:
            if sink not in execute_time:
                continue
            ub = max(ub, execute_time[sink])

        MUL[e] = [lb, ub]

    return MUL  


def long_lived_first(graph, order, gcd):
    if type(order) == str:
        order = get_order(graph, order)

    mem_loc = {}
    MUL = extract_spans(graph, order)

    live_tree = IntervalTree()
    for e, span in MUL.items():
        lb, ub = MUL[e]
        live_tree[lb : ub + 1] = e

    ava_offsets = defaultdict(lambda: [])
    num_timesteps = len(graph.nodes.values())
    ava_offsets[0] = [(1, num_timesteps)]

    n_tensors = len(graph.edges.values())
    size_zero = 0
    for e in graph.edges.values():
        if e.size == 0 or e.source not in order:
            size_zero += 1

    traverse = set()
    while len(traverse) < n_tensors - size_zero and len(ava_offsets) > 0:
        # Get lowest offset.
        min_offset = 99999999999999
        for offset, _ in ava_offsets.items():
            if len(ava_offsets[offset]) == 0:
                continue
            
            min_offset = min(min_offset, offset)

        ava_gaps = ava_offsets[min_offset]
        for gap in ava_gaps:
            ava_lb, ava_ub = gap
            ava_offsets[min_offset].remove(gap)

            # Get the longest-lived tensor.
            interval = Interval(ava_lb, ava_ub + 1)
            overlaps = list(live_tree.overlap(interval))
            longest_tensor = None
            max_liveness = 0
            for overlap in overlaps:
                lb, ub, tensor = overlap[0], overlap[1], overlap[2]
                if lb < ava_lb or ub > ava_ub + 1:
                    continue

                if tensor.size == 0 or tensor in traverse:
                    continue
                
                len_live = ub - lb
                if len_live > max_liveness or \
                    (len_live == max_liveness and longest_tensor != None and tensor.name < longest_tensor.name):
                    longest_tensor = tensor
                    max_liveness = ub - lb

            if longest_tensor:
                # Split current offset.
                traverse.add(longest_tensor)
                
                mem_loc[longest_tensor] = min_offset
                
                if len(traverse) == n_tensors - size_zero:
                    break

                left, right = MUL[longest_tensor]
                new_offset = min_offset + (longest_tensor.size // gcd) * gcd
                ava_offsets[new_offset].append((left, right))
                
                if ava_lb <= left - 1:
                    ava_offsets[min_offset].append((ava_lb, left - 1))
                if right + 1 <= ava_ub: 
                    ava_offsets[min_offset].append((right + 1, ava_ub))
                
            else:
                if len(traverse) == n_tensors - size_zero:
                    break
                    
                left_offset, left_gap = sys.maxsize, None
                right_offset, right_gap = sys.maxsize, None
                
                for offset, intervals in ava_offsets.items(): 
                    for interval in intervals:
                        lb, ub = interval
                        
                        # Get left offset.
                        if ub == ava_lb - 1:
                            left_offset = offset
                            left_gap = interval
                        
                        # Get right offset.
                        if lb == ava_ub + 1:
                            right_offset = offset
                            right_gap = interval
                        
                if left_gap == None and right_gap == None:
                    import pdb;pdb.set_trace()

                if right_gap is None or left_offset < right_offset:
                    new_lb = left_gap[0]      # lb of left_merge.
                    new_ub = ava_ub
                    ava_offsets[left_offset].append((new_lb, new_ub))
                    ava_offsets[left_offset].remove(left_gap)
                elif left_gap is None or left_offset > right_offset:
                    new_lb = ava_lb
                    new_ub = right_gap[1]     # ub of right_merge.
                    ava_offsets[right_offset].append((new_lb, new_ub))
                    ava_offsets[right_offset].remove(right_gap)
                else:
                    assert(left_offset == right_offset)
                    new_lb = left_gap[0]      # lb of left_merge.
                    new_ub = right_gap[1]     # ub of right_merge.
                    ava_offsets[left_offset].append((new_lb, new_ub))
                    ava_offsets[left_offset].remove(left_gap)
                    ava_offsets[left_offset].remove(right_gap)
                
    peak_memory = 0
    for e, address in mem_loc.items():
        top = address + (e.size // gcd) * gcd
        peak_memory = max(peak_memory, top)

    schedule = defaultdict(lambda: [[], []])    
    new_mem_loc = defaultdict(lambda: {})
    for e in graph.edges.values():
        if e.size == 0 or e.source not in order:
            continue

        allocate, deallocate = MUL[e]
        s0 = [str(allocate) + "@" + str(mem_loc[e])]
        s1 = []
        for t in range(allocate + 1, deallocate + 1):
            s1.append(t)
        schedule[e] = [s0, s1]

        for t in range(allocate, deallocate + 1):
            new_mem_loc[t][e] = mem_loc[e]

    graph_name = graph.name
    model = graph_name.split("_")[0]
    visualizer.draw_schedule(schedule, img_path=f"./logs/graphs/{model}/" + graph.name + "_long_lived_first" + ".png")
    return peak_memory, new_mem_loc