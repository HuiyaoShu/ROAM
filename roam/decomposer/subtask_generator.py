
class subtask_generator:
    def __init__(
        self,
        graph,
        key_points,
        spans,
        all_activations,
        NUM_CONS_NODES = 200,
        NUM_CONS_EDGES = 30
    ):
        self.graph = graph
        self.key_points = key_points
        self.spans = spans
        self.all_activations = all_activations
        self.NUM_CONS_NODES = NUM_CONS_NODES
        self.NUM_CONS_EDGES = NUM_CONS_EDGES
        self.final_time_keypoints = {}

    def is_fwd(self, node):
        CONS, cons = 10, 0
        BFS = [node]
        while cons <= CONS:
            n = BFS.pop(0)
            for e in node.fanout:
                if e in self.all_activations:
                    return True
                for sink in e.sinks:
                    if sink not in BFS:
                        BFS.append(sink)
            cons += 1

        return False

    def get_most_inner(self):
        num_key_points = len(self.key_points)

        fwd, bwd = None, None
        index = (num_key_points + 1) // 2 - 1
        while (not fwd) and (not bwd):
            _current = self.key_points[index]
            if self.is_fwd(_current):
                if index > num_key_points - 1:
                    print("Error: No backward subgraph.")
                    return None

                _next = self.key_points[index + 1]
                if not self.is_fwd(_next):
                    fwd = index
                    bwd = index + 1
                else:
                    index += 1

            if not self.is_fwd(_current):
                if index - 1 < 0:
                    print("Error: No forward subgraph. ")
                    return None

                _pre = self.key_points[index - 1]
                if self.is_fwd(_pre):
                    fwd = index - 1
                    bwd = index
                else:
                    index -= 1

        return fwd, bwd

    def is_subtask(self, fwd_graph, bwd_graph):
        f_node, f_inner = fwd_graph[0], fwd_graph[1]
        b_node, b_inner = bwd_graph[0], bwd_graph[1]

        bound_1, bound_2, bound_3, bound_4 = 9999999,9999999,-9999999,9999999
        asap, alap, MUL = self.spans[0], self.spans[1], self.spans[2]
        bound_1 = asap[f_node]
        bound_4 = asap[b_node]
        if f_inner:
            bound_2 = asap[f_inner]
        if b_inner:
            bound_3 = asap[b_inner]
        # import pdb;pdb.set_trace()

        out_in_bwd = True
        input_in_fwd = True

        traverse = []
        BFS = [f_node, b_node]
        while len(BFS) != 0:
            node = BFS.pop(0)
            traverse.append(node)
            if bound_1 <= asap[node] and alap[node] <= bound_2:
                if node != b_node and node != f_inner:
                    for e in node.fanout:
                        if MUL[e][1] > bound_4:
                            out_in_bwd = False
                        for sink in e.sinks:
                            if alap[sink] <= bound_2 and alap[sink] <= bound_4 and sink not in traverse:
                                BFS.append(sink)

            if bound_3 <= asap[node] and alap[node] <= bound_4:
                if node != f_node and node != b_inner:
                    for e in node.fanin:
                        if MUL[e][0] < bound_1:
                            input_in_fwd = False
                        source = e.source
                        if asap[source] >= bound_3 and asap[source] >= bound_1 and source not in traverse:
                            BFS.append(source)

        return out_in_bwd, input_in_fwd, len(traverse)

    def merge_subtask(self, subtasks):
        num_tasks = len(subtasks)
        _start = 0
        while num_tasks >= 2:
            # Whether can merge the two tasks.
            num_nodes, num_edges = 0, 0
            for k, v in subtasks[_start].items():
                task1 = k
                num_nodes += v[0]
                num_edges += v[1]
            if num_nodes >= self.NUM_CONS_NODES: # or num_edges >= self.NUM_CONS_EDGES:
                _start += 1
                continue

            evict_ones = [_start, ]
            for i in range(_start + 1, num_tasks):
                for k, v in subtasks[i].items():
                    task2 = list(k)
                    num_nodes += v[0]
                    num_edges += v[1]

                if num_nodes < self.NUM_CONS_NODES and num_edges < self.NUM_CONS_EDGES:
                    task1[0] = task2[0]
                    task1[3] = task2[3]
                    evict_ones.append(i)
                else:
                    _start = i

            for index in evict_ones:
                subtasks.pop(index)
            subtasks.insert(0, {tuple(task1) : [num_nodes, num_edges]})

        return subtasks

    def generate_subtask(self):
        # (f_node, f_inner, b_inner, b_node) => [num_nodes, num_edges]
        subtasks = []
        num_key_points = len(self.key_points)

        # Get first fwd, bwd subgraph.
        fwd, bwd = self.get_most_inner()
        center = -1
        for i in range(num_key_points):
            if "sum_1" in self.key_points[i].name:
                center = i

        last_fwd, last_bwd = center, center

        while fwd >= 0 and bwd < num_key_points:
            f_node = self.key_points[fwd]
            b_node = self.key_points[bwd]
            f_inner = self.key_points[last_fwd] if last_fwd else None
            b_inner = self.key_points[last_bwd] if last_bwd else None

            fwd_graph = [f_node, f_inner]
            bwd_graph = [b_node, b_inner]

            out_in_bwd, input_in_fwd, num_nodes = self.is_subtask(fwd_graph, bwd_graph)
            if out_in_bwd & input_in_fwd:
                last_fwd, last_bwd = fwd, bwd
                fwd -= 1
                bwd += 1
                task_info = {(f_node, f_inner, b_inner, b_node): [num_nodes]}
                subtasks.append(task_info)

            if not out_in_bwd:
                bwd += 1
            if not input_in_fwd:
                fwd -= 1

        return subtasks

    def _get_nodes(self, fwd, inner_fwd, inner_bwd, bwd):
        fwd_node = self.key_points[fwd] if fwd and fwd >= 0 else None
        inner_fwd_node = self.key_points[inner_fwd]
        inner_bwd_node = self.key_points[inner_bwd]
        bwd_node = self.key_points[bwd] if bwd and bwd < len(self.key_points) else None

        asap, alap = self.spans[0], self.spans[1]
        bound1 = asap[fwd_node] if fwd_node else -999999
        bound2, bound3 = asap[inner_fwd_node], asap[inner_bwd_node]
        bound4 = asap[bwd_node] if bwd_node else 999999

        # Inner nodes:
        # [1] bound1<=asap<=alap<=bound2
        # [2] bound3<=asap<=alap<=bound4
        inner_nodes = set()
        BFS = [inner_fwd_node, inner_bwd_node]
        while len(BFS) != 0:
            node = BFS.pop(0)
            inner_nodes.add(node)

            for e in node.fanout:
                for sink in e.sinks:
                    if sink in inner_nodes or sink in BFS:
                        continue

                    if (bound1 <= asap[sink] and alap[sink] <= bound2) or \
                        (bound3 <= asap[sink] and alap[sink] <= bound4):
                        BFS.append(sink)

            for e in node.fanin:
                source = e.source
                if source in inner_nodes or sink in BFS:
                    continue

                if (bound1 <= asap[source] and alap[source] <= bound2) or \
                    (bound3 <= asap[source] and alap[source] <= bound4):
                    BFS.append(source)

        return inner_nodes

    def _split_bwd(self, fwd, inner_fwd, inner_bwd, bwd, previous):
        inner_nodes = self._get_nodes(fwd, inner_fwd, inner_bwd, bwd)
        num_nodes = len(inner_nodes)

        if num_nodes > self.NUM_CONS_NODES or bwd >= len(self.key_points):
            return previous if previous else \
                ([fwd, inner_fwd, inner_bwd, bwd], num_nodes)
        else:
            previous = ([fwd, inner_fwd, inner_bwd, bwd], num_nodes)
            return self._split_bwd(None, inner_fwd, inner_bwd, bwd + 1, previous)

    def _merge_tasks(self, fwd, inner_fwd, inner_bwd, bwd, previous):
        inner_nodes = self._get_nodes(fwd, inner_fwd, inner_bwd, bwd)
        num_nodes = len(inner_nodes)

        if num_nodes > self.NUM_CONS_NODES or fwd < 0 or bwd >= len(self.key_points):
            return previous if previous else \
                ([fwd, inner_fwd, inner_bwd, bwd], num_nodes)
        else:
            previous = ([fwd, inner_fwd, inner_bwd, bwd], num_nodes)
            return self._merge_tasks(fwd - 1, inner_fwd, inner_bwd, bwd + 1, previous)

    def get_sum_1(self):
        for i in range(len(self.key_points)):
            if self.key_points[i].name == "sum_1":
                return i

    def get_task(self, task_id):
        task_node = []
        for i in range(4):
            if task_id[i] != None:
                task_node.append(self.key_points[task_id[i]])
            else:
                task_node.append(None)
        return task_node

    def generate_subtask_flexible(self):
        # import pdb;pdb.set_trace()
        subtasks = []
        key_points = self.key_points
        num_key_points = len(self.key_points)
        asap = self.spans[0]

        sum_1 = self.get_sum_1()
        fwd, bwd = sum_1 - 1, sum_1 + 1
        inner_fwd, inner_bwd = sum_1, sum_1
        last_task = None

        while fwd >= 0 and bwd < num_key_points:
            task = self._merge_tasks(fwd, inner_fwd, inner_bwd, bwd, previous=None)
            if task == None:
                break

            last_task = task
            inner_fwd, inner_bwd = task[0][0], task[0][-1]
            fwd = inner_fwd - 1
            bwd = inner_bwd + 1

            task_node = self.get_task(task[0])

            # subtasks.append((task_node, task[1]))
            subtasks.append({tuple(task_node): task[1]})

            for i in range(4):
                n = task_node[i]
                if n:
                    self.final_time_keypoints[asap[n]] = n   


        # Further split backward subgraph.
        # outer_task = (None, inner_fwd, inner_bwd, None)
        num_nodes = self._get_nodes(None, inner_fwd, inner_bwd, None)
        if len(num_nodes) > self.NUM_CONS_NODES and inner_bwd < len(key_points) - 1:
            bwd = inner_bwd + 1
            while(bwd < len(key_points)):
                task = self._split_bwd(None, inner_fwd, inner_bwd, bwd, previous=None)
                inner_bwd = task[0][-1]
                bwd = inner_bwd + 1
                last_task = task

                if task == None:
                    return subtasks

                task_node = self.get_task(task[0])
                subtasks.append({tuple(task_node): task[1]})

                for i in range(4):
                    n = task_node[i]
                    if n:
                        self.final_time_keypoints[asap[n]] = n   

        # print("[***] Subtasks after split for backward subgraph: {}".format(subtasks))

        outer_task = []
        if last_task and last_task[0][0] != None:
            outer_task.append(None)
            outer_task.append(last_task[0][0])
            outer_task.append(last_task[0][-1])
            outer_task.append(None)

            task_node = self.get_task(outer_task)
            num_nodes = len(self._get_nodes(None, outer_task[1], outer_task[2], None))
            subtasks.append({tuple(task_node): num_nodes})

            self.final_time_keypoints.update({asap[task_node[1]]: task_node[1], \
                                                asap[task_node[2]]: task_node[2]})

        print("[***] All subtasks({}): {}".format(len(subtasks), subtasks))
        return subtasks