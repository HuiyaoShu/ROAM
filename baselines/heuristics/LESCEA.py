import sys
import json
import torch
import torchtext
import torchvision
from rota.tools import utils, visualizer
from collections import defaultdict
from intervaltree import IntervalTree, Interval

def list_memory_scheduler(graph):
    # import pdb;pdb.set_trace()
    order = {}
    ready_list = set()

    # Initialize ready list.
    num_nodes = len(graph.nodes.values())
    for n in graph.nodes.values():
        if len(n.fanin) == 0:
            ready_list.add(n)
    
    timestep = 1
    base_memory = 0
    peak_memory = 0
    while len(ready_list) > 0 and len(order) < num_nodes:
        # Choose a node to execute.
        freed_current = 0
        defined_current = 0

        chosen_node_1 = None       # Get according to peak memory.
        min_live = sys.maxsize
        free_1, define_1 = 0, 0
        chosen_node_2 = None       # Get according to live memory.
        min_peak = sys.maxsize
        free_2, define_2 = 0, 0
        exceed_peak = True
        for node in ready_list:
            freed_bytes = 0
            for e in node.fanin:
                flag = True
                for sink in e.sinks:
                    if sink != node and sink not in order:
                        flag = False
                        break
                if flag:
                    freed_bytes += e.size
            
            defined_bytes = 0
            for e in node.fanout:
                defined_bytes += e.size    
            
            # If all exceed the previous peak memory,
            # then choose the node generate the smallest peak memory.
            if base_memory + defined_bytes > peak_memory:
                tmp1 = base_memory + defined_bytes
                # if base_memory + defined_bytes < min_peak:
                # if tmp1 < min_peak or \
                #     (tmp1 == min_peak and chosen_node_1 != None and node.name < chosen_node_1.name)
                if tmp1 <= min_peak:
                    if tmp1 == min_peak and (chosen_node_1 == None or node.name > chosen_node_1.name):
                        continue
                    chosen_node_1 = node
                    min_peak = base_memory + defined_bytes
                    free_1, define_1 = freed_bytes, defined_bytes
                
                continue
            
            # If there is one or more than one node generate peak memory the previous one,
            # then choose the node generate the smallest live memory.
            tmp2 = base_memory + defined_bytes - freed_bytes
            # if base_memory + defined_bytes - freed_bytes < min_live:
            # if tmp2 < min_live or \
            #     (tmp2 == min_live and chosen_node_2 != None and node.name < chosen_node_2.name)
            if tmp2 <= min_live:
                if tmp2 == min_live and (chosen_node_2 != None and node.name > chosen_node_2.name):
                    continue
                exceed_peak = False
                chosen_node_2 = node
                min_live = base_memory + defined_bytes - freed_bytes 
                free_2, define_2 = freed_bytes, defined_bytes


        chosen_node = None
        free, define = 0, 0
        if exceed_peak:
            chosen_node = chosen_node_1
            free, define = free_1, define_1
        else:
            chosen_node = chosen_node_2
            free, define = free_2, define_2

        if chosen_node:
            peak_memory = max(peak_memory, base_memory + define)    # During execution.
            base_memory = base_memory + define - free               # After execution.
            order[chosen_node] = timestep
            # print("Executing node {} at {}.".format(chosen_node, timestep))
            timestep += 1

            # Update ready_list.
            ready_list.remove(chosen_node)
            for e in chosen_node.fanout:
                for sink in e.sinks:
                    if len(sink.fanin) == 1:
                        ready_list.add(sink)
                        continue
                    
                    flag = True
                    for e in sink.fanin:
                        if e.source not in order:
                            flag = False
                            break
                    
                    if flag:
                        ready_list.add(sink)
    
    return order, peak_memory