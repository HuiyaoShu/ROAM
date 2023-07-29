
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from inspect import currentframe

import intervaltree
import math
from collections import defaultdict
from rota.tools import dataflow_graph

def _GCD(values):
    if len(values) == 0:
        return 1
    if len(values) == 1:
        return values[0]
    elif len(values) == 2:
        return math.gcd(values[0], values[1])
    else:
        middle = len(values) // 2
        return math.gcd(_GCD(values[:middle]), _GCD(values[middle:]))


def get_model_size(model, model_name):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    # all_size = (param_size + buffer_size) / 1024 / 1024
    all_size = param_size / 1024 / 1024
    print('[+] Model {} size: {:.3f}MB'.format(model_name, all_size))
    return all_size


def run_simulation(graph, node_ordering):
    edge_ref_counts = defaultdict(lambda: 0)

    memory_used = 0
    peak_memory = 0
    mem_per_timestep = []
    for n, t in node_ordering:
        for fanout in n.fanout:
            if fanout.size > 0:
                # edge_ref_counts: number of tensors users.
                edge_ref_counts[fanout] = len(fanout.sinks)
                memory_used += fanout.size

        if memory_used > peak_memory:
            peak_memory = memory_used
        mem_per_timestep.append((n, memory_used))

        for fanin in n.fanin:
            if fanin.size == 0:
                continue
            edge_ref_counts[fanin] -= 1
            if edge_ref_counts[fanin] < 0:
                import pdb;pdb.set_trace()
            assert edge_ref_counts[fanin] >= 0
            if edge_ref_counts[fanin] == 0:
                memory_used -= fanin.size

    return (peak_memory, mem_per_timestep)


def get_linenumber():
    cf = currentframe()
    return cf.f_back.f_lineno


def extract_node_ordering(graph, schedule):
    node_ordering = {}
    for t, s in schedule.items():
        generated = parse_schedule_item(s[0][0])
        if t.source in node_ordering:
            # assert node_ordering[t.source] == generated
            # fanouts are not genrated at the same time.
            if node_ordering[t.source] != generated:
                print("Confilct for {}: ordering {} vs generate {}".format(t.source, node_ordering[t.source], generated))
        else:
            node_ordering[t.source] = generated

    # Special case for nodes that have no fanout
    for n in graph.nodes.values():
        if len(n.fanout) > 0:
            continue
        if n.is_stateful():
            continue
        available = set()
        for i in range(0, len(n.fanin)):
            s = schedule[n.fanin[i]]

            fanin_availability = set(s[1])
            for generate in s[0]:
                fanin_availability.add(parse_schedule_item(generate) + 1)
            if i == 0:
                available = fanin_availability
            else:
                available.intersection_update(fanin_availability)

        if not available:
            continue
        else:
            node_ordering[n] = sorted(available)[0]

    return node_ordering


def transfer_order(order):
    re_order = {}
    for n, t in order.items():
        re_order[n.name] = t

    return re_order


# Make sure that tensors never overlap in time
def validate_address_allocation(memory_locations, verbose=True):
    tol = 1
    for ts, pairs in memory_locations.items():
        used_addresses = intervaltree.IntervalTree()
        for tensor, address in pairs.items():
            if tensor.size == 0:
                continue
            if used_addresses.overlaps(address, address + tensor.size):
                intervals = used_addresses[address : address + tensor.size]
                for obj in intervals:
                    delta1 = address + tensor.size - obj[0]
                    if delta1 >= 0 and delta1 <= tol:
                        continue
                    delta2 = obj[1] - address
                    if delta2 >= 0 and delta2 <= tol:
                        continue
            used_addresses[address : address + tensor.size] = tensor

    return True


def parse_schedule_item(item):
    item = str(item)
    item = item.split("[")[0]
    items = item.split("@")
    assert len(items) > 0
    return int(items[0])


def check_op_inputs(tensor, allocate, schedule, verbose=True):
    assert tensor.source is not None
    op = tensor.source
    for fanin in op.fanin:
        preserve = schedule[fanin][1]
        spills = [parse_schedule_item(i) for i in schedule[fanin][2]]
        for t in allocate:
            if t not in preserve and t not in spills:
                if verbose:
                    print(
                        f"Input tensor {fanin.name} not in memory when op {op.name} driving {tensor.name} is run at time {t}"
                    )
    return True


def validate_timeline(schedule, verbose=True):
    for tensor, s in schedule.items():
        if isinstance(tensor, dataflow_graph.Node):
            continue

        # Create time.
        allocate = [parse_schedule_item(i) for i in s[0]]

        if not check_op_inputs(tensor, allocate, schedule, verbose):
            return False

        if tensor.size == 0:
            continue
        preserve = s[1]
        if len(preserve) == 0:
            continue

        spills = [parse_schedule_item(i) for i in s[2]]

        if preserve[0] - 1 not in allocate: # and preserve[0] - 1 not in spills:
            print(
                f"tensor {tensor.name} was not allocated/fetched before timestep {preserve[0]}"
            )
            # return False

        for i in range(1, len(preserve) - 1):
            if preserve[i] == preserve[i - 1] + 1:
                continue
            if preserve[i] - 1 in spills:
                continue
            if preserve[i] - 1 in allocate:
                continue

            print(
                f"tensor {tensor.name} was not allocated/fetched/preserved before timestep {preserve[i]}"
            )
            # return False

    return True

def validate_node_ordering(graph, schedule, verbose=True):
    node_order = extract_node_ordering(graph, schedule)
    order = []
    node_per_step = {}
    timesteps = []

    # print("[***] Node ordering: {}".format(node_order))
    for node, ts in node_order.items():
        timesteps.append(ts)
        if ts in node_per_step:
            node_per_step[ts] += 1
        else:
            node_per_step[ts] = 1
            
        order.append((ts, node))
    
    timesteps.sort()

    order.sort(key=lambda obj: obj[0])
    already_executed = set()
    for _, node in order:
        for fi in node.fanin:
            for src in fi.sources:
                if src not in already_executed:
                    print(
                        f"Invalid order: {node.name} scheduled before its fanin {src.name}"
                    )
                    print(f"Complete ordering {[node.name for _, node in order]}")
                    # return False
        already_executed.add(node)
        
    return True, node_order

def validate_tensors(graph, overall_schedule):
    res = True
    
    traverse = set()
    for e, _ in overall_schedule.items():
        if e.name in traverse:
            res = False
            print("Tensor {} appears in overall_schedule more than one time.".format(e))
        traverse.add(e.name)

    for e in graph.edges.values():
        if e.name not in traverse:
            res = False
            print("Tensor {} are missed in overall_schedule.".format(e))
    
    return res