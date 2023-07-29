# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import math
import sys
from collections import defaultdict, OrderedDict

import intervaltree
from intervaltree import IntervalTree, Interval
from rota.lao_and_reducer import ilp_solver
from rota.tools import utils
from rota.tools import dataflow_graph

class Scheduler:
    def __init__(
        self,
        graph,
        subtask,
        gcd,
        mode=None,
        timeout_s=None,
        ratio=1000,
        rel_stop=None,
        solver="GUROBI",
        timestep_factor=1.0,
        print_relaxation=False,
    ):
        self.graph = graph
        self.subtask = subtask
        self.gcd = gcd
        self.mode = mode
        self.timeout = timeout_s
        self.ratio = ratio
        self.rel_stop = rel_stop
        self.solver = solver
        self.timestep_factor = timestep_factor
        self.print_relaxation = print_relaxation
    

    class TimeStepsForNode:
        def __init__(self, node, spans, startoffset=0):
            asap = spans[0]
            alap = spans[1]
            lb = asap[node]
            ub = alap[node]
            assert not node.is_stateful()
            self.iter = iter(range(lb + startoffset, ub + 1))

        def __iter__(self):
            return self

        def __next__(self):
            return self.iter.__next__()

    class TimeStepsForEdge:
        def __init__(self, edge, spans, bounds, attr, startoffset=0):
            # if edge.name == "getitem_499:0_opt":
            #     import pdb;pdb.set_trace()

            lb, ub = spans[2][edge]

            intervals = intervaltree.IntervalTree()
            for snk in edge.sinks:
                if snk.op_type == "stateful_node_sink":
                    continue
                new_lb = spans[0][snk]
                new_lb = max(0, new_lb - 1)
                new_ub = spans[1][snk]
                intervals.add(intervaltree.Interval(new_lb, new_ub + 1))

            if not edge.is_stateful():
                src = edge.source
                new_lb = spans[0][src]
                new_ub = spans[1][src]
                intervals.add(intervaltree.Interval(new_lb, new_ub + 1))
            else:
                intervals.add(intervaltree.Interval(lb, lb + 1))
                intervals.add(intervaltree.Interval(ub, ub + 1))

            ts = []
            intervals.merge_overlaps()
            for i in intervals:
                ts += list(range(i.begin, i.end))

            ts = []
            intervals.merge_overlaps()
            for i in intervals:
                ts += list(range(i.begin, i.end))

            # CIFO
            if attr == 1:
                if ub > bounds[3]:
                    ts.append(bounds[3])
                elif ub > bounds[1]:
                    ts.append(bounds[1])

            # COFI
            if attr == 2:
                if lb < bounds[0]:
                    ts.append(bounds[0])
                elif lb < bounds[2]:
                    ts.append(bounds[2])
                
            ts = list(set(ts))
            ts = sorted(ts)
            self.res = ts
            self.iter = iter(ts[startoffset:])

        def __add__(self, other):
            return list(set(self.res)&set(other.res))

        def __iter__(self):
            return self

        def __next__(self):
            return self.iter.__next__()

    class TimeStepsForFanin:
        def __init__(self, node, spans, bounds, attrs):
            if len(node.fanin) == 0:
                self.iter = iter([])
                return

            attr = attrs[node.fanin[0]]
            timesteps = set(Scheduler.TimeStepsForEdge(node.fanin[0], spans, bounds, attr))
            for i in range(1, len(node.fanin)):
                attr = attrs[node.fanin[i]]
                timesteps &= set(Scheduler.TimeStepsForEdge(node.fanin[i], spans, bounds, attr))

            timesteps = sorted(timesteps)
            self.iter = iter(timesteps)

        def __iter__(self):
            return self

        def __next__(self):
            return self.iter.__next__()

    class DensePreserveVarsMap:
        def __init__(self, sparse_map):
            local_map = {}
            maxi = 0
            mini = math.inf
            for i in sparse_map:
                if i > maxi:
                    maxi = i
                if i < mini:
                    mini = i
            for i in range(maxi, mini - 1, -1):
                assert i >= mini
                assert i <= maxi
                if i not in sparse_map:
                    local_map[i] = local_map[i + 1]
                else:
                    local_map[i] = sparse_map[i]
            self.local_map = OrderedDict(sorted(local_map.items()))
            # Return full map of preserve vars: t => 0 or 1.

        def __getitem__(self, index):
            return self.local_map[index]

        def items(self):
            return self.local_map.items()

    class DenseGenerateOrFetchVarsMap:
        def __init__(self, sparse_map):
            self.sparse_map = sparse_map

        def __getitem__(self, index):
            if index not in self.sparse_map:
                return 0
            return self.sparse_map[index]

    def ComputeMinMaxMemoryRequired(self, all_tensors, nodes):
        min_memory_requirement = 0
        for n in nodes:
            mem_needed = 0
            for e in n.fanin:
                if e not in all_tensors:
                    continue
                mem_needed += e.size
            for e in n.fanout:
                if e not in all_tensors:
                    continue
                mem_needed += e.size
            if mem_needed < min_memory_requirement:
                min_memory_requirement = mem_needed
        
        max_memory_requirement = 0
        for e in all_tensors:
            max_memory_requirement += e.size
        return min_memory_requirement, max_memory_requirement

    def update_spans(self, asap, alap, MUL, schedule_constraints):
        for vertex, t in schedule_constraints.items():
            asap[vertex] = t
            alap[vertex] = t
            for e in vertex.fanout:
                MUL[e][0] = t
            for e in vertex.fanin:
                max_alap = -1
                for sink in e.sinks:
                    max_alap = max(max_alap, alap[sink])
                MUL[e][1] = max_alap
        
        return [asap, alap, MUL]

    def update_alap(self, asap, alap, schedule_constraints):
        for vertex, t in schedule_constraints.items():
            alap[vertex] = t
        return alap

    def update_MUL(self, MUL, asap, alap, all_tensors):
        for e in all_tensors:
            src = e.source
            lb = asap[src]
            ub = -1
            for sink in e.sinks:
                ub = max(ub, alap[sink])
            MUL[e] = [lb, ub]
        return MUL        
        
    def GetSolver(
            self,
            generate_addresses=True,
            user_schedule=None
        ):
        print("[***] RUNNING SOLVER: GENERATE_ADDRESS = {}".format(generate_addresses))

        graph, g_fwd, g_bwd, spans = self.subtask.get_info()
        fwd, inner_fwd = g_fwd[0], g_fwd[1]
        bwd, inner_bwd = g_bwd[0], g_bwd[1]

        gcd = self.gcd
        full_live_full_add, sub_live_full_add, sub_live_no_add, activations, nodes = self.subtask.get_tensors_nodes()
        # all_tensors = list(full_live_full_add) + list(sub_live_full_add)
        all_tensors = list(full_live_full_add) + list(sub_live_full_add) + list(sub_live_no_add)
        min_memory_requirement, max_memory_requirement = self.ComputeMinMaxMemoryRequired(all_tensors, list(nodes))
        max_address = max_memory_requirement // gcd

        int_feas_tol = None
        # if defrag or account_for_fragmentation:
        if max_address > 0:
            int_feas_tol = min(1e-5, 1.0 / max_address)
            int_feas_tol = max(1e-9, int_feas_tol)
        
            if 1.0 / max_address > 1e-5:
                print(f"Tightened IntFeasTol to {int_feas_tol}")

        solver = ilp_solver.ILPSolver(
            timeout_s=self.timeout,
            rel_stop=self.rel_stop,
            solver=self.solver,
            int_feas_tol=int_feas_tol,
            extra_params={"MIPFocus": 2},
        )

        all_tensors = list(full_live_full_add) + list(sub_live_full_add) + list(sub_live_no_add)
        address_tensors = list(full_live_full_add) + list(sub_live_full_add)


        # method: subtask.info
        asap, alap, MUL = spans[0], spans[1], spans[2]
        
        schedule_constraints = {}
        if user_schedule is not None:
            for n, ts in user_schedule.items():
                # import pdb;pdb.set_trace()
                if ts <= 0:
                    raise ValueError(
                        "Negative or null timestep specified for node %s" % str(n)
                    )
                if isinstance(n, str):
                    name = n
                else:
                    # assert isinstance(n, dataflow_graph.Node)
                    name = n.name
                if name not in graph.nodes:
                    print(
                        "Invalid schedule: node "
                        + name
                        + " does not exist in the graph"
                    )
                    continue
                n = graph.nodes[name]
                schedule_constraints[n] = ts
            
            if schedule_constraints and generate_addresses:
                spans = self.update_spans(asap, alap, MUL, schedule_constraints)
                MUL = self.update_MUL(MUL, asap, alap, all_tensors)
                spans[2] = MUL

        # print("[+] Number of tensors: {}".format(len(MUL)))

        bound1 = asap[fwd] if fwd else 1
        bound2 = asap[inner_fwd]
        bound3 = asap[inner_bwd]
        bound4 = asap[bwd] if bwd else len(graph.nodes.values())
        bounds = [bound1, bound2, bound3, bound4]

        generate_vars = defaultdict(lambda: {})
        preserve_vars = defaultdict(lambda: {})
        create_at_timestep = defaultdict(lambda: 0)
        tensor_attribute = self.subtask.tensor_attribute

        if not generate_addresses:
            num_binary_vars = 0
            num_interger_vars = 0
            all_tensors = list(set(all_tensors))
            address_tensors = []
            for e in all_tensors:
                # Create variables: C, P
                tt = 0
                attr = tensor_attribute[e]
                for t in self.TimeStepsForEdge(e, spans, bounds, attr):
                # for t in timesteps:
                    v = solver.create_binary_var(e.name + "generate_ts" + str(t))
                    generate_vars[e][t] = v

                    # if tensor_attribute[e] <= 1:
                    create_at_timestep[t] += v

                    v = solver.create_binary_var(e.name + "preserve_ts" + str(t))
                    preserve_vars[e][t] = v

            max_fanout = 1
            for n in graph.nodes.values():
                max_fanout = max(max_fanout, len(n.fanout))

            # print("Max fanout: {}".format(max_fanout))

            # Add constraints for single stream.
            for e in all_tensors:
                # "CIFO"
                if tensor_attribute[e] == 1:
                    _, free = MUL[e]
                    
                    if free > bound4:
                        solver.add_constraint(
                            preserve_vars[e][bound4] == 1,
                            name=f"{utils.get_linenumber()}_preserve_{e.name}_at_bound4_{bound4}",
                        )
                    elif free > bound2:
                        solver.add_constraint(
                            preserve_vars[e][bound2] == 1,
                            name=f"{utils.get_linenumber()}_preserve_{e.name}_at_bound2_{bound2}",
                        )
                # "COFI"
                elif tensor_attribute[e] == 2:
                    create, _ = MUL[e]

                    if create < bound1:
                        solver.add_constraint(
                            preserve_vars[e][bound1] == 1,
                            name=f"{utils.get_linenumber()}_preserve_{e.name}_at_bound1_{bound1}",
                        )
                    elif create < bound3:
                        solver.add_constraint(
                            preserve_vars[e][bound3] == 1,
                            name=f"{utils.get_linenumber()}_preserve_{e.name}_at_bound3_{bound3}",
                        )

            all_timestep = []
            # all_timestep.append(bound1)         # None, inner_fwd, ...
            for i in range(bound1, bound2):
                all_timestep.append(i)
            
            for i in range(bound3, bound4):
                all_timestep.append(i)
            
            processed = set()
            for n in nodes:
                if asap[n] == alap[n]:
                    t = asap[n]
                    if t > all_timestep[-1]:
                        continue

                    processed.add(t)
                    num_fanout = 0
                    for e in n.fanout:
                        if e not in all_tensors:
                            continue
                        num_fanout += 1

                        solver.add_constraint(
                            generate_vars[e][t] == 1,
                            name=f"{utils.get_linenumber()}_generated_{e.name}_created_{n.name}_at_{t}",
                        )
                        
                        solver.add_constraint(
                            preserve_vars[e][t] == 0,
                            name=f"{utils.get_linenumber()}_generated_{e.name}_created_{n.name}_at_{t}",
                        )

            if self.mode == "ss":
                # Only for single stream: Add single stream constraints.
                print("Process single stream...")
                for t in all_timestep:
                    if t in processed:
                        continue

                    solver.add_constraint(
                        create_at_timestep[t] >= 1,
                        name=f"{utils.get_linenumber()}_create_at_least_one_at_{t}",
                    )
                    
                    solver.add_constraint(
                        create_at_timestep[t] <= max_fanout,
                        name=f"{utils.get_linenumber()}_create_at_least_one_at_{t}",
                    )


            for e in all_tensors:
                attr = tensor_attribute[e]
                prev = self.TimeStepsForEdge(e, spans, bounds, attr)
                cur = self.TimeStepsForEdge(e, spans, bounds, attr, startoffset=1)
                for t in cur:
                    p = prev.__next__()

                    solver.add_constraint(
                        preserve_vars[e][t]
                        <= preserve_vars[e][p] + generate_vars[e][p],
                        name=f"{utils.get_linenumber()}_{e.name}_precedence@{t}",
                    )
                    solver.add_constraint(
                        preserve_vars[e][t] + generate_vars[e][t] <= 1,
                        name=f"{utils.get_linenumber()}_{e.name}_at_most_one@{t}",
                    )
                
                lb, ub = MUL[e]
                if e.source in schedule_constraints:
                    ts = schedule_constraints[e.source]
                    assert ts >= lb
                    assert ts <= ub
                    attr = tensor_attribute[e]
                    for t in self.TimeStepsForEdge(e, spans, bounds, attr):
                        if t != ts:
                            solver.add_constraint(
                                generate_vars[e][t] == 0,
                                name=f"{utils.get_linenumber()}_{e.name}_generate_var@{t}",
                            )
                        else:
                            solver.add_constraint(
                                generate_vars[e][t] == 1,
                                name=f"{utils.get_linenumber()}_{e.name}_generate_var@{t}",
                            )

                for snk in e.sinks:
                    if snk in schedule_constraints:
                        ts = schedule_constraints[snk]
                        assert ts >= lb
                        assert ts <= ub
                        solver.add_constraint(
                            preserve_vars[e][ts] == 1,
                            name=f"{utils.get_linenumber()}_{e.name}_preserve_var@{ts}",
                        )

                lb, _ = MUL[e]
                solver.add_constraint(
                    preserve_vars[e][lb] == 0,
                    name=f"{utils.get_linenumber()}_{e.name}_preserve_var@{lb}",
                )

                attr = tensor_attribute[e]
                for t in self.TimeStepsForEdge(e, spans, bounds, attr):
                    if t > alap[e.source]:
                        # if not allow_rematerialization or e.is_stateful():
                        solver.add_constraint(
                            generate_vars[e][t] == 0,
                            name=f"{utils.get_linenumber()}_{e.name}_generate_var_past_alap@{t}",
                        )

                # Multi-stream: make sure the latter subgrap runnnig after the previous subgraph.
                src = e.source
                if src == fwd or src == inner_bwd:
                    for t in self.TimeStepsForEdge(e, spans, bounds, attr):
                        if t < alap[src]:
                            solver.add_constraint(
                                generate_vars[e][t] == 0,
                                name=f"{utils.get_linenumber()}_{e.name}_generate_var_earlier_alap_key_points@{t}"
                            )

                if e.size == 0:
                    prev = self.TimeStepsForEdge(e, spans, bounds, attr)
                    for t in self.TimeStepsForEdge(e, spans, bounds, attr, startoffset=1):
                        p = prev.__next__()
                        if t <= alap[e.source]:
                            solver.add_constraint(
                                preserve_vars[e][t]
                                >= preserve_vars[e][p] + generate_vars[e][p],
                                name=f"{utils.get_linenumber()}_{e.name}_ctrl_edge@{t}",
                            )
                        else:
                            # The source node has been run at this point.
                            solver.add_constraint(
                                preserve_vars[e][t] == 1,
                                name=f"{utils.get_linenumber()}_{e.name}_preserve_var@{t}",
                            )

            for e, ts in generate_vars.items():
                s = 0
                for v in ts.values():
                    s += v

                solver.add_constraint(
                    1 == s,
                    name=f"{utils.get_linenumber()}_{e.name}_materialized_once",
                )    

            precedence_nodes = set()
            for n in nodes:
                if len(n.fanout) > 1:
                    first_e = None
                    for i in range(0, len(n.fanout)):
                        if n.fanout[i] not in all_tensors:
                            continue
                        
                        if first_e is None:
                            first_e = n.fanout[i]
                            continue                        

                        for t in self.TimeStepsForNode(n, spans): 
                            solver.add_constraint(
                                generate_vars[first_e][t]
                                == generate_vars[n.fanout[i]][t],
                                name=f"{utils.get_linenumber()}_{n.name}_all_fanouts_generated@{t}",
                            )

                
                for snk in n.fanout:
                    for src in n.fanin:
                        # Add precedence constraints
                        if src not in all_tensors or snk not in all_tensors:
                            continue
                        for t in self.TimeStepsForNode(n, spans):
                            solver.add_constraint(
                                generate_vars[snk][t]
                                <= preserve_vars[src][t],
                                name=f"{utils.get_linenumber()}_{snk.name}_{src.name}_precedence@{t}",
                            )
                            precedence_nodes.add(n)
            
            # A node needs to consume all its inputs at the same timestep. Handle
            # the case where the node has no fanout below. The case where the node
            # has fanout is already handled (in precedence constraints).
            for n in nodes:
                if len(n.fanout) > 0 and n != g_fwd[0] and n != g_bwd[0]: # and n in precedence_nodes:
                    continue
                if len(n.fanin) <= 1:
                    continue

                # We need at least one timestep during which all the inputs are live at the same time
                sum_of_all_live = 0
                attrs = {}
                for e in n.fanin:
                    if e not in tensor_attribute:
                        attrs[e] = -1
                        continue
                    
                    attrs[e] = tensor_attribute[e]
                
                for t in self.TimeStepsForFanin(n, spans, bounds, attrs):
                    if t > alap[n]:
                        break
                    #if n.name == "threshold_backward_2" and t == 39:
                    #    import pdb;pdb.set_trace()
                    all_live = solver.create_binary_var(
                        "fanin_of_" + n.name + "_live_at_ts" + str(t)
                    )

                    for f in n.fanin:
                        if f not in all_tensors:
                            continue
                        solver.add_constraint(
                            # Solver tends to obtain minimized all_live, if 0 or 1, then must 0.
                            all_live <= preserve_vars[f][t],
                            name=f"{utils.get_linenumber()}_fanin_of_{n.name}_dead_due_to_{f.name}@{t}",
                        )
                    sum_of_all_live += all_live
                solver.add_constraint(
                    sum_of_all_live >= 1,
                    name=f"{utils.get_linenumber()}_fanin_of_{n.name}_live_at_one_ts",
                )

            # Preserved in memory once it is created until freed.
            for e in all_tensors:
                first = alap[e.source] + 1
                last = 0
                for v in e.sinks:
                    last = max(last, asap[v])

                attr = tensor_attribute[e]
                for ts in self.TimeStepsForEdge(e, spans, bounds, attr):
                    if ts < first or ts > last:
                        continue
                    solver.add_constraint(
                        preserve_vars[e][ts] == 1,
                        name=f"{utils.get_linenumber()}_{e.name}_must_be_preserved_@{t}",
                    )

        # Tensors do not include params
        max_address = max_memory_requirement // gcd
        min_address = min_memory_requirement // gcd
        max_address_for_atv = sum([atv.size for atv in activations]) // gcd
        
        if not generate_addresses:
            # Memory usage at each timestep
            mem_at_timestep = defaultdict(lambda: 0)
            for e in all_tensors:
                if e.size == 0:
                    continue

                # import pdb;pdb.set_trace()
                for t, v in self.DensePreserveVarsMap(preserve_vars[e]).items():
                    mem_at_timestep[t] += v * (e.size // gcd)

                for t, v in generate_vars[e].items():
                    mem_at_timestep[t] += v * (e.size // gcd)
            
            # Add constraints for memory at each timestep.
            liveness = defaultdict(lambda: [])
            for e in address_tensors:
                if e.size == 0:
                    continue
                # lb, ub = makespan[e]
                lb, ub = MUL[e]
                for ts in range(lb, ub + 1):
                    liveness[ts].append(e)
            
            for ts, mem_usage in mem_at_timestep.items():
                max_mem = 0
                for e in liveness[ts]:
                    max_mem += e.size
                solver.add_constraint(
                    mem_usage <= max_address,
                    name=f"{utils.get_linenumber()}_mem_usage_less_than_{max_address}@{ts}"
                )
                solver.add_constraint(
                    mem_usage >= min_address,
                    name=f"{utils.get_linenumber()}_mem_usage_less_than_{max_address}@{ts}"
                )
    
        if generate_addresses: 
            for e in all_tensors:
                lb, ub = spans[2][e]
                attr = tensor_attribute[e]
                for t in self.TimeStepsForEdge(e, spans, bounds, attr):
                    if t < lb or t > ub:
                        continue

                    v = solver.create_binary_var(e.name + "generate_ts" + str(t))
                    generate_vars[e][t] = v

                    v = solver.create_binary_var(e.name + "preserve_ts" + str(t))
                    preserve_vars[e][t] = v

            for e in all_tensors:
                first = True
                lb, ub = spans[2][e]
                attr = tensor_attribute[e]
                for t in self.TimeStepsForEdge(e, spans, bounds, attr):
                    if t < lb or t > ub:
                        continue
                    
                    if first:
                        solver.add_constraint(
                            generate_vars[e][t] == 1,
                            name=f"{utils.get_linenumber()}_{e.name}_preserve_var@{t}",
                        )
                        solver.add_constraint(
                            preserve_vars[e][t] == 0,
                            name=f"{utils.get_linenumber()}_{e.name}_preserve_var@{t}",
                        )
                        first = False
                    else:
                        solver.add_constraint(
                            generate_vars[e][t] == 0,
                            name=f"{utils.get_linenumber()}_{e.name}_preserve_var@{t}",
                        )
                        solver.add_constraint(
                            preserve_vars[e][t] == 1,
                            name=f"{utils.get_linenumber()}_{e.name}_preserve_var@{t}",
                        )

            addresses = OrderedDict()
            for tensor in address_tensors:
                if tensor in activations:
                    v = solver.create_integer_var(
                        tensor.name,
                        lower_bound=0,
                        upper_bound=max_address_for_atv - tensor.size // gcd 
                    )
                    addresses[tensor] = v
                    # num_interger_vars += 1
                else:
                    v = solver.create_integer_var(
                        tensor.name,
                        lower_bound=0,
                        upper_bound=max_address - tensor.size // gcd,
                    )
                    addresses[tensor] = v

            # Tensor assignment.
            fixed_locations = {}
            processed = set()
            base_address = 0
            tensor_free = {}
            for tensor in address_tensors:
                if tensor not in activations:
                    continue
                tensor_free[tensor] = MUL[tensor][1]
            sorted_tensor_free = sorted(tensor_free.items(), key=lambda x:x[1], reverse=True)
            
            mem_used = intervaltree.IntervalTree()
            for tensor, _ in sorted_tensor_free:
                solver.add_constraint(
                    addresses[tensor] == base_address,
                    name=f"{utils.get_linenumber()}_force_{tensor.name}_at_{base_address}",
                )
                fixed_locations[tensor] = base_address

                base_address += tensor.size // gcd
                start = MUL[tensor][0]
                end = MUL[tensor][1]
                processed.add(tensor)
                mem_used[start : end + 1] = base_address

            max_mem = base_address
            for t, a in addresses.items():
                if t in fixed_locations:
                    a.Start = fixed_locations[t]
                else:
                    span = MUL[t]
                    max_address_used = 0
                    # print(f"Querying intervaltree {span[0]} {span[1]+1}")
                    for interval in mem_used.overlap(span[0], span[1] + 1):
                        # print(f"address {interval.data} used")
                        max_address_used = max(max_address_used, interval.data)
                    a.Start = max_address_used
                    # print(f"Adding gen address to intervaltree {span[0]} {span[1]}")
                    if span[0] >= span[1] + 1:
                        import pdb;pdb.set_trace()
                    mem_used[span[0] : span[1] + 1] = (
                        max_address_used + t.size // gcd
                    )
                    max_mem = max(max_mem, max_address_used + t.size // gcd)

            for t, a in addresses.items():
                if t not in fixed_locations:
                    solver.add_constraint(
                        a <= max_mem - t.size // gcd,
                        name=f"{utils.get_linenumber()}_tighten_max_address_for_{t.name}",
                    )

            processed = set()
            for t1, span1 in MUL.items():
                if t1 not in address_tensors or t1.size == 0:
                    continue
                
                for t2, span2 in MUL.items():
                    if t2  not in address_tensors or t2.size == 0:
                        continue
                    if t1 is t2 or (t2, t1) in processed:
                        continue

                    processed.add((t1, t2))
                    if (
                        span1[1] < span2[0]
                        or span1[0] > span2[1]
                        or not graph.can_overlap_in_time(t1, t2)   # Can multiplex.
                    ):
                        # print(t1.name + " and " + t2.name + "CANNOT OVERLAP")
                        continue

                    if t1 in fixed_locations and t2 in fixed_locations:
                        continue

                    live_together = graph.are_connected_by_node(t1, t2)    # Can multiplex.
                    if not live_together:
                        if alap[t1.source] <= asap[t2.source]:
                            for snk in t1.sinks:
                                if asap[snk] >= alap[t2.source]:
                                    live_together = True
                        elif alap[t2.source] <= asap[t1.source]:
                            for snk in t2.sinks:
                                if asap[snk] >= alap[t1.source]:
                                    live_together = True

                    if live_together:
                        if t1 in fixed_locations:
                            solver.add_constraint(
                                addresses[t2] >= fixed_locations[t1] + t1.size // gcd,
                                name=f"{utils.get_linenumber()}_force_{t1.name}_below_{t2.name}",
                            )

                        elif t2 in fixed_locations:
                            solver.add_constraint(
                                addresses[t1] >= fixed_locations[t2] + t2.size // gcd,
                                name=f"{utils.get_linenumber()}_force_{t1.name}_above_{t2.name}",
                            )
                        else:
                            v = solver.create_binary_var(
                                name=f"{utils.get_linenumber()}_{t1.name}_below_{t2.name}"
                            )

                            solver.add_constraint(
                                addresses[t1] + t1.size // gcd - addresses[t2]
                                <= (1 - v) * max_address,
                                name=f"{utils.get_linenumber()}_force_{t1.name}_below_{t2.name}",
                            )
                            solver.add_constraint(
                                addresses[t1] - addresses[t2] - t2.size // gcd
                                >= -v * max_address,
                                name=f"{utils.get_linenumber()}_force_{t1.name}_above_{t2.name}",
                            )

                    elif span1[1] >= span2[0] and span1[0] <= span2[1]:
                        v1 = solver.create_binary_var(t1.name + "_" + t2.name + "_v1")
                        solver.add_constraint(
                            addresses[t1] + t1.size // gcd - addresses[t2]
                            <= (1 - v1) * max_address,
                            name=f"{utils.get_linenumber()}_{t1.name}_below_{t2.name}",
                        )

                        v2 = solver.create_binary_var(t1.name + "_" + t2.name + "_v2")
                        solver.add_constraint(
                            addresses[t1] - addresses[t2] - t2.size // gcd
                            >= (v2 - 1) * max_address,
                            name=f"{utils.get_linenumber()}_{t1.name}_above_{t2.name}",
                        )
                       
                        if t1 in fixed_locations:
                            solver.add_constraint(v1 == 1)
                        elif t2 in fixed_locations:
                            solver.add_constraint(v2 == 1)
                        else:
                            # Let's check if that helps
                            solver.add_constraint(v1 + v2 <= 1)

                        # check if they actually do overlap
                        generate_t1 = self.DenseGenerateOrFetchVarsMap(
                            generate_vars[t1]
                        )
                        preserve_t1 = self.DensePreserveVarsMap(preserve_vars[t1])
                        generate_t2 = self.DenseGenerateOrFetchVarsMap(
                            generate_vars[t2]
                        )
                        preserve_t2 = self.DensePreserveVarsMap(preserve_vars[t2])

                        for ts in range(
                            max(span1[0], span2[0]), min(span1[1], span2[1]) + 1
                        ):
                            live1 = generate_t1[ts] + preserve_t1[ts] 
                            live2 = generate_t2[ts] + preserve_t2[ts] 
                            overlap_at_t = live1 + live2 - 1
                            solver.add_constraint(
                                v1 + v2 >= overlap_at_t,
                                name=f"{utils.get_linenumber()}_{t1.name}_overlaps_{t2.name}@{ts}",
                            )

        max_usage = (max_memory_requirement) // gcd

        s = 0
        v = solver.create_integer_var(
                "peak_memory_usage",
                lower_bound=min_memory_requirement // gcd,
                upper_bound=max_usage,
        )
        s += v

        if not generate_addresses:
            # print("[***] Max usage: {}".format(max_usage))
            for ts, m in mem_at_timestep.items():
                solver.add_constraint(
                    v >= m, name=f"{utils.get_linenumber()}_max_address@{ts}"
                )
        else:
            for t, a in addresses.items():
                solver.add_constraint(
                    v >= a + t.size // gcd,
                    name=f"{utils.get_linenumber()}_max_address_above_{t.name}",
                )

        solver.set_objective_function(s, maximize=False)

        result = solver.solve()
        
        '''
        1. Sink with no fanout: last_use=alap[sink]
        2. Sink with fanout: last_use=max(C[sink.fanout])
        '''
        last_uses = {}
        for e in all_tensors:
            last_use = 0
            for sink in e.sinks:
                if len(sink.fanout) == 0 or sink == g_fwd[1] or sink == g_bwd[0]:
                    last_use = max(last_use, alap[sink])
                else:
                    for fanout in sink.fanout:
                        for t, v in generate_vars[fanout].items():
                            if result[v] >= 0.99:
                                last_use = max(last_use, t)

            last_uses[e] = last_use

        schedule = defaultdict(lambda: ([], [], [], []))
        mem_locations = defaultdict(lambda: {})
        materialization_count = {}
        tensor_attribute = self.subtask.tensor_attribute
        for n, ts in generate_vars.items():
            tensor_materialization_count = 0
            for t, v in ts.items():
                if result[v] >= 0.99:
                    tensor_materialization_count += 1
                    if n.size == 0:
                        schedule[n][0].append(str(t) + "@[ctrl]")
                    elif n.is_stateful() and (not spilling_allowed):
                        schedule[n][0].append(str(t) + "@[weight]")
                    else:
                        if n in address_tensors:
                            offset = int(result[addresses[n]] * gcd)
                        else:
                            offset = -1     # If not in address_tensors, then offset equals -1.

                        schedule[n][0].append(
                            str(t) + "@" + str(offset)
                        )
                        schedule[n][2].append(t)
                        mem_locations[t][n] = offset 
                    # Import CIFI/CIFO/COFI information:
                    # CIFI: 0
                    # CIFO: 1
                    # COFI: 2
                    # bound_tensors: 3
                    schedule[n][3].append(tensor_attribute[n])

        for n, ts in preserve_vars.items():
            for t, v in self.DensePreserveVarsMap(ts).items():
                if t > last_uses[n]:
                    continue
                if result[v] >= 0.99:
                    schedule[n][1].append(t)
                    if n.size == 0:
                        continue
                    if n in address_tensors:
                        offset = int(result[addresses[n]] * gcd)
                    else:
                        offset = -1
                    mem_locations[t][n] = offset
            if len(schedule[n][1]) >= 1:
                schedule[n][2].append(schedule[n][1][-1])
            else:
                if schedule[n][0][0].endswith("[ctrl]"):
                    continue
                schedule[n][2].append(int(schedule[n][0][0].split("@")[0]))
              
        
        # Get required memory for stage 1.
        mem_at_timestep = defaultdict(lambda: 0)
        for e, ts in preserve_vars.items():
            if e not in full_live_full_add and e not in sub_live_full_add:
                continue

            for t, v in self.DensePreserveVarsMap(ts).items():
                if t <= last_uses[e]:
                    mem_at_timestep[t] += result[v] * e.size
        for e, ts in generate_vars.items():
            if e not in full_live_full_add and e not in sub_live_full_add:
                continue

            for t, v in ts.items():
                if t <= last_uses[e]:
                    mem_at_timestep[t] += result[v] * e.size
        
        peak_mem_usage = 0
        for mem_usage in mem_at_timestep.values():
            peak_mem_usage = max(peak_mem_usage, mem_usage)
            
        
        required_memory = 0
        if not generate_addresses:
            required_memory = peak_mem_usage
        else:
            for t, a in addresses.items():
                if t.size == 0:
                    continue
                elif t.is_stateful() and (not spilling_allowed):
                    continue
                required_memory = max(required_memory, result[a] * gcd + t.size)
        
        print("[+] Required memory: {}".format(required_memory))
        summary = {
            "required_memory": required_memory,
        }

        node_order = None
        if not generate_addresses:
            node_order = utils.extract_node_ordering(graph, schedule)

        return summary, schedule, node_order, mem_locations


    def is_overlap(self, lv1, lv2):
        if lv2[0] >= lv1[0] and lv2[0] <= lv1[1]:
            return True
        elif lv2[1] >= lv1[0] and lv2[1] <= lv1[1]:
            return True
        
        return False
    

    def assignment_for_tensors(
        self, 
        shared_tensors, 
        overall_schedule, 
        tensor_liveness,
        all_activations,
    ):
        gcd = self.gcd
        all_tensors = self.graph.edges.values()
        nodes = self.graph.nodes.values()
        min_memory_requirement, max_memory_requirement = self.ComputeMinMaxMemoryRequired(all_tensors, nodes)
        max_address = max_memory_requirement // gcd

        int_feas_tol = None
        # if defrag or account_for_fragmentation:
        int_feas_tol = min(1e-5, 1.0 / max_address)
        int_feas_tol = max(1e-9, int_feas_tol)
        if 1.0 / max_address > 1e-5:
            print(f"Tightened IntFeasTol to {int_feas_tol}")

        solver = ilp_solver.ILPSolver(
            timeout_s=self.timeout,
            rel_stop=self.rel_stop,
            solver=self.solver,
            int_feas_tol=int_feas_tol,
            extra_params={"MIPFocus": 1},
        ) 

        addresses = OrderedDict()
        for tensor, s in shared_tensors.items():    # t1
            if tensor.size == 0:
                continue
            # To obtain max_address
            v = solver.create_integer_var(
                tensor.name,
                lower_bound=0,
                upper_bound=max_address - tensor.size // gcd,
            )
            addresses[tensor] = v 

        processed = set()
        for tensor, s in shared_tensors.items():
            allocate = int(s[0][0].split("@")[0])
            deallocate = utils.parse_schedule_item(s[1][-1]) if len(s[1]) > 0 else allocate
            
            # May introduce unnecessary overlap constraints.
            if allocate == deallocate:
                alive_interval = Interval(allocate - 1, deallocate + 1)
            else:
                alive_interval = Interval(allocate, deallocate + 1)
            
            overlaps = tensor_liveness.overlap(alive_interval)
            for interval in overlaps:
                overlap_tensor = interval[2]       # t2
                if overlap_tensor == tensor:
                    continue
                
                if overlap_tensor in shared_tensors:
                    overlap_s = shared_tensors[overlap_tensor]
                else:
                    overlap_s = overall_schedule[overlap_tensor]

                address = int(overlap_s[0][0].split("@")[1])
                t1, t2 = tensor, overlap_tensor
                
                if (t2.name, t1.name) in processed:
                    continue
                
                processed.add((t1.name, t2.name))
                # add constraints.
                if t2 in shared_tensors:
                    v = solver.create_binary_var(
                        name=f"{utils.get_linenumber()}_{t1.name}_below_{t2.name}"
                    )
                    solver.add_constraint(
                        addresses[t1] + t1.size // gcd - addresses[t2]
                        <= (1 - v) * max_address,
                        name=f"{utils.get_linenumber()}_force_{t1.name}_below_{t2.name}",
                    )
                    solver.add_constraint(
                        addresses[t1] - addresses[t2] - t2.size // gcd
                        >= -v * max_address,
                        name=f"{utils.get_linenumber()}_force_{t1.name}_above_{t2.name}",
                    )
                else:
                    solver.add_constraint(
                        addresses[t1] >= address // gcd + t2.size // gcd,
                        name=f"{utils.get_linenumber()}_force_{t1.name}_above_{t2.name}",
                    )
        
        s = 0
        v = solver.create_integer_var(
                "peak_memory_usage",
                lower_bound=min_memory_requirement // gcd,
                upper_bound=max_memory_requirement // gcd,
        )
        s += v
        for t, a in addresses.items():
            solver.add_constraint(
                v >= a + t.size // gcd,
                name=f"{utils.get_linenumber()}_max_address_above_{t.name}",
            )

        solver.set_objective_function(s, maximize=False)

        result = solver.solve()

        for t, a in addresses.items():
            old_s = shared_tensors[t]
            allocate = old_s[0][0].split("@")[0]
            old_s[0][0] = str(allocate) + "@" + str(int(result[a] * gcd))
            overall_schedule[t] = old_s
        
        overall_schedule = self.adjust_tmp_buffer(overall_schedule, tensor_liveness, all_activations)
        return overall_schedule
    

    def adjust_tmp_buffer(self, overall_schedule, tensor_liveness, all_activations):
        gcd = self.gcd

        # busy time ==> offset
        liveness_address = IntervalTree()
        used_address = defaultdict(lambda: IntervalTree())
        locations = {}
        for interval in tensor_liveness.items():
            allocate = interval[0]
            deallocate = interval[1]
            tensor = interval[2]
            s = overall_schedule[tensor]
            if s[0][0].endswith("[ctrl]"):
                continue
            address = int(s[0][0].split("@")[1])
            locations[tensor] = address
            top = address + (tensor.size // gcd) * gcd

            liveness_address[allocate : deallocate] = top

            for t in range(allocate, deallocate):
                if t not in used_address:
                    used_address[t] = IntervalTree()
                used_address[t][address : top] = tensor
        
        # Sort for tensor locations.
        sorted_locations = sorted(locations.items(), key=lambda x:x[1])

        for e, _ in sorted_locations:
            if "pt" in self.graph.name and  e in all_activations:
                continue

            s = overall_schedule[e]

            if s[0][0].endswith("[ctrl]"):
                continue

            address = int(s[0][0].split("@")[1])
            allocate = int(s[2][0])
            deallocate = int(s[2][1])
            interval = Interval(allocate, deallocate + 1)
            
            move_to = sys.maxsize
            overlaps = list(liveness_address.overlap(interval))
            for interval in overlaps:
                offset = int(interval[2])
                if offset >= move_to:  
                    continue

                new_range = Interval(offset, offset + e.size)
                
                flag = True
                for t in range(allocate, deallocate + 1):
                    overlap_address = list(used_address[t].overlap(new_range))
                    if len(overlap_address) == 0 or (len(overlap_address) == 1 and overlap_address[0][2] == e):
                        continue
                    else:
                        flag = False
                        break
                
                if flag:
                    move_to = offset

            if move_to < address:
                new_offset = move_to
                s[0][0] = str(allocate) + "@" + str(new_offset)
                
                old_top = address + (e.size // gcd) * gcd
                new_top = new_offset + (e.size // gcd) * gcd
                liveness_address.removei(allocate, deallocate + 1, old_top)
                liveness_address[allocate : deallocate + 1] = new_top
                for t in range(allocate, deallocate + 1):
                    used_address[t].removei(address, old_top, e)
                    used_address[t][new_offset : new_top] = e
                # print("[info] Adjust tensor {} address: {} -> {}.".format(e.name, str(address), str(new_offset)))

            overall_schedule[e] = s

        return overall_schedule

    
    # STEP-1: Reduce the liveness of tensors.
    # STEP-2: Merge blocks.
    def reduce(self, schedules, all_activations, tensor_schedule_map):
        # Travrese schedules of all subtasks.
        overall_schedule = {}
        base = 0
        used_address = IntervalTree()
        tensor_liveness = IntervalTree()
        shared_tensors = {}

        for e, sids in tensor_schedule_map.items():
            if len(sids) <= 1:
                schedules[sids[0]][e][3].append(sids[0])
                continue
            schedule_first = schedules[sids[0]]

            min_create = 999999
            max_free = -1
            address_id = -1
            address = -1
            overall_attr = -1
            for id in sids:
                s = schedules[id][e]
                items = s[0][0].split("@")
                assert(len(items) == 2)

                attribute = int(s[3][0])
                # CIFI(0) or CIFO(1)
                if attribute <= 1:# or attribute == 3:
                    min_create = min(min_create, int(items[0]))
                
                # CIFI(0) or COFI(2)
                if len(s[1]) > 0 and (attribute % 2 == 0 or attribute == 3):
                    max_free = max(max_free, int(s[1][-1]))
                
                if items[1].endswith("[ctrl]"):
                    address = items[1]
                elif int(items[1]) != -1:
                    address_id = id
                    address = int(items[1])
                    overall_attr = attribute
                else:
                    schedules[id].pop(e)
            
            if max_free == -1:
                max_free = min_create

            if type(address) == int:
                if address < 0:
                    print("[***] Address leq 0: {}".format(e))

            new_s = [[], [], [], []]
            new_s[0].append(str(min_create) + "@" + str(address))
            for i in range(min_create + 1, max_free + 1):
                new_s[1].append(i)
            new_s[2].append(min_create)
            new_s[2].append(max_free)
            new_s[3].append(overall_attr)
            new_s[3].append(address_id)
            schedules[address_id][e] = new_s
            
        for i in range(len(schedules)):
            schedule = schedules[i]
            current_base = base
            for tensor, s in schedule.items():
                assert(len(s[0]) == 1)
                if s[0][0].endswith("[ctrl]"):
                    continue

                items = s[0][0].split("@")
                assert(len(items) == 2)
                allocate = int(items[0])
                deallocate = utils.parse_schedule_item(s[1][-1]) if len(s[1]) > 0 else allocate

                tensor_liveness[allocate : deallocate + 1] = tensor
                address = int(items[1])
                new_address = current_base + address
                s[0][0] = str(allocate) + "@" + str(new_address)

                # address overlap and liveness overlap.
                tmp_interval = Interval(new_address, new_address + tensor.size)
                overlaps = list(used_address.overlap(tmp_interval))
                
                # Sort overlaps according to tensor sizes (large -> small).
                for i in range(1, len(overlaps)):
                    for j in range(0, len(overlaps) - i):
                        t1 = overlaps[j][2][0]
                        t2 = overlaps[j+1][2][0]
                        tmp = overlaps[j]
                        if t1.size < t2.size:
                            overlaps[j] = overlaps[j+1]
                            overlaps[j+1] = tmp

                flag = True
                for interval in overlaps:
                    tmp_item = interval[2]
                    t, t_s = tmp_item[0], tmp_item[1]
                    
                    # Whether overlapping on liveness between the two tensors.
                    if self.is_overlap(s[2], t_s[2]):
                        # print("[+] overlap address: {}, {} \nvs\n {}, {}".format(tensor, s, t, t_s))
                        moved_tensor = None
                        # Preserve current tensor if current tensor is activation 
                        # or current tensor is larger than overlap tensor.

                        # pri = alpha * (tensor.size / t.size) + \
                        #    beta * ((int(s[2][1]) - int(s[2][0])) / (int(t_s[2][1]) - int(t_s[2][0])))

                        # if t.size < tensor.size or tensor in all_activations:
                        # To optimize.
                        if t.size < tensor.size or (tensor in all_activations and t.size / tensor.size < self.ratio):
                        # if t.size < tensor.size or tensor in all_activations:
                            moved_tensor = t
                            sch = t_s
                            used_address.remove(interval)
                        else:
                            flag = False
                            moved_tensor = tensor
                            sch = s
                        
                        shared_tensors[moved_tensor] = sch
                        if not flag:
                            break
                if flag:
                    used_address[new_address + 1 : new_address + tensor.size] = [tensor, s]

                overall_schedule.update(schedule)
                
                if tensor in all_activations:
                    base = max(base, new_address + (tensor.size // self.gcd) * self.gcd)
        
        # print("shared_tensors: {}".format(shared_tensors))
        overall_schedule = self.assignment_for_tensors(shared_tensors, overall_schedule, tensor_liveness, all_activations)

        required_memory = 0
        mem_locations = defaultdict(lambda: {})
        for tensor, s in overall_schedule.items():
            if s[0][0].endswith("[ctrl]"):
                continue
            tmp_s = s[0][0].split("@")
            allocate = int(tmp_s[0])
            address = int(tmp_s[1])
            if address == -1:
                continue

            required_memory = max(required_memory, address + tensor.size)

            mem_locations[allocate][tensor] = address
            for t in s[1]:
                mem_locations[int(t)][tensor] = address

        return overall_schedule, required_memory, mem_locations