from roam.tools import utils
from roam.decomposer import subtask_info
from roam.decomposer.subtask_generator import subtask_generator
from roam.decomposer.key_points import compute_spans, find_key_points, get_all_activations
from roam.lao_and_reducer.subtask_solver import Scheduler

def generate_schedulers(model_name, graph, mode, NL, TL, ratio):
    # Get key_points.
    key_points, asap, alap, MUL = find_key_points(graph)
    print("Key_points({}): {}".format(len(key_points), key_points))

    # Get all activations.
    spans = [asap, alap, MUL]
    all_activations = get_all_activations(graph, spans)


    # Get subtasks.
    print("Number of nodes for each subtask: {}".format(NL))
    sg = subtask_generator(graph, key_points, spans, all_activations, NL)
    subtasks = sg.generate_subtask_flexible()

    ##############################
    # Adjust zero-fanin ctrl edges.
    adjust_edges = []
    apply_edges = []
    for e in graph.edges.values():
        if "_zero_fanin_forced_late" in e.name:
            adjust_edges.append(e)
        if "_forced_early" in e.name:
            apply_edges.append(e)

    final_time_keypoints = sg.final_time_keypoints
    timesteps = [k for k, _ in final_time_keypoints.items()]
    timesteps = sorted(timesteps)

    if mode == "ms" or model_name == "efficientnet":
        for e in adjust_edges:
            if "pt" in model_name:
                continue

            lb, _ = MUL[e][0], MUL[e][1]
            n = e.sinks[0]
            
            # Get the nearest pre keypoint for subtasks.
            pre_bound = None
            for i in range(len(timesteps)):
                if lb - timesteps[i] < 0:
                    pre_bound = final_time_keypoints[timesteps[i - 1]]
                    break
            
            if pre_bound == None or pre_bound == e.source:
                continue

            graph.delete_edge(e)
            graph.add_edge(
                [pre_bound],
                [n],
                size=0,
                name=n.name + "_zero_fanin_forced_late",
            )

        graph.canonical = False
        graph.canonicalize()
    #################################
    
    # Generate information of subtasks.
    key_points, asap, alap, MUL = find_key_points(graph)
    spans = [asap, alap, MUL]
    solver_spans = compute_spans(graph) if mode == "ms" else None

    i = 1
    num_subtask = len(subtasks)
    tasks_info = []
    for task in subtasks:
        for k, _ in task.items():
            tasks_info.append(subtask_info.Subtask(graph, num_subtask - i, \
                k[0], k[1], k[2], k[3], spans, solver_spans, all_activations))
        i += 1

    # Create scheulers.
    schedulers = []
    tensor_sizes = [e.size for e in graph.edges.values()]
    gcd = utils._GCD(tensor_sizes)
    if "pt" in graph.name:
        rel_stop = None
    else:
        rel_stop = 1e-6

    for task in tasks_info:
        schedulers.append(Scheduler(None, task, gcd, mode=mode, rel_stop=rel_stop, timeout_s=TL, ratio=ratio, print_relaxation=True,))

    global_scheduler = Scheduler(
        graph,
        None, 
        gcd,
        None,
        rel_stop=rel_stop, 
        timeout_s=TL, 
        print_relaxation=True,
    )

    return global_scheduler, schedulers, all_activations