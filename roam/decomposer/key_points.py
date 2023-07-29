import math

def compute_asap_serial(graph):
    asap = {}
    for node in graph.nodes.values():
        is_visited = []
        # path = _compute_asap_serial(node, asap, is_visited)
        path = _compute_asap_serial_test(node)
        asap[node] = path
    return asap


def _compute_asap_serial_test(node):
    BFS = [node]
    traverse = set()
    while len(BFS) != 0:
        n = BFS.pop(0)
        if n in traverse:
            continue
        traverse.add(n)

        for e in n.fanin:
            src = e.source
            if src not in BFS:
                BFS.append(src)
    
    return len(traverse)


def _compute_asap_serial(node, asap, is_visited):
    if len(node.fanin) == 0:
        is_count = 1 if node not in is_visited else 0
        is_visited.append(node)
        return is_count
    if node in is_visited:
        return 0

    path = 0
    for e in node.fanin:
        source = e.source
        path += _compute_asap_serial(source, asap, is_visited)

    path += 1

    is_visited.append(node)
    return path


def compute_alap_serial(graph, max_timesteps):
    alap = {}
    for node in graph.nodes.values():
        is_visited = []
        path = _compute_alap_serial_test(node)
        alap[node] = max_timesteps - path + 1
    return alap


def _compute_alap_serial_test(node):
    BFS = [node]
    traverse = set()
    while len(BFS) != 0:
        n = BFS.pop(0)
        if n in traverse:
            continue
        traverse.add(n)

        for e in n.fanout:
            for sink in e.sinks:
                if sink not in BFS:
                    BFS.append(sink)
    
    return len(traverse)


def _compute_alap_serial(node, alap, is_visited, max_timesteps):
    if len(node.fanout) == 0 and node not in is_visited:
        is_visited.append(node)
        return 1
    elif node in is_visited:
        return 0

    path = 0
    for e in node.fanout:
        for sink in e.sinks:
            path += _compute_alap_serial(sink, alap, is_visited, max_timesteps)

    path += 1

    is_visited.append(node)
    return path


def compute_asap_parallel(graph):
    timings = {}
    for vertex in graph.nodes.values():
        _compute_asap_parallel(vertex, timings)
    return timings


def _compute_asap_parallel(vertex, timings):
    if vertex in timings:
            # Already visited
            return timings[vertex]
    time = 1
    for e in vertex.fanin:
        source = e.source
        t = _compute_asap_parallel(source, timings)
        time = max(t + 1, time)

    timings[vertex] = time

    return time


def compute_alap_parallel(graph, max_timesteps):
    timings = {}
    for vertex in graph.nodes.values():
        _compute_alap_parallel(
            vertex, timings, max_timesteps
        )
    return timings


def _compute_alap_parallel(
    vertex, timings, max_timesteps
):
    if vertex in timings:
        # Already visited
        return timings[vertex]

    time = max_timesteps
    for e in vertex.fanout:
        for sink in e.sinks:
            t = _compute_alap_parallel(
                sink, timings, max_timesteps
            )
            time = min(t - 1, time)

    timings[vertex] = time
    return time


def make_span(graph, asap, alap):
    MUL = {}
    for e in graph.edges.values():
        left, right = 0, 0
        source = e.source
        left = asap[source]
        for sink in e.sinks:
            right = max(right, alap[sink])
        MUL[e] = [left, right]
    return MUL


def get_all_activations(graph, spans):
    all_activations = []
    loss_node = graph.find_node("sum_1")    # How to generate activations more generally?
    loss_node_t = spans[0][loss_node]
    MUL = spans[2]
    for e in graph.edges.values():
        if e.is_stateful():
            continue
        lb, ub = MUL[e][0], MUL[e][1]
        if lb < loss_node_t and ub > loss_node_t:
            all_activations.append(e)
    
    return all_activations


def _sort(key_points, asap):
        length = len(key_points)
        for i in range(1, length):
            for j in range(0, length - i):
                node1 = key_points[j]
                node2 = key_points[j + 1]
                if asap[node1] > asap[node2]:
                    key_points[j] = node2
                    key_points[j+1] = node1

        return key_points   


def find_key_points(graph):
    max_timesteps = len(graph.nodes.values())
    asap = compute_asap_serial(graph)
    alap = compute_alap_serial(graph, max_timesteps)
    MUL = make_span(graph, asap, alap)

    key_points = []
    for node in graph.nodes.values():
        if alap[node] - asap[node] == 0:
            key_points.append(node)

    key_points = _sort(key_points, asap)

    return key_points, asap, alap, MUL


def compute_spans(graph):
    num_timesteps = len(graph.nodes.values())
    longest_path_length = graph.longest_path_length()
    num_timesteps = min(
        int(math.ceil(longest_path_length * 1.01)), num_timesteps
    )
    timestep_factor = 1
    num_timesteps = int(num_timesteps * timestep_factor)

    if timestep_factor < 1 and longest_path_length > num_timesteps:
        print(
            f"Adjusting num_timesteps to {longest_path_length} to ensure there are enough steps to run the longest path"
        )
        num_timesteps = longest_path_length
    
    asap = compute_asap_parallel(graph)
    
    num_timesteps = longest_path_length
    alap = compute_alap_parallel(graph, num_timesteps)
    MUL = make_span(graph, asap, alap)

    return [asap, alap, MUL]