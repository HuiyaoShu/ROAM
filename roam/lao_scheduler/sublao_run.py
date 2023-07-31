from roam.tools import visualizer
from dill import pickle, dumps, loads

def run_scheduler(input_scheduler, id, model_name, graph_name, mode, schedules):
    print("[+] Running process {}".format(id))
    
    scheduler = loads(input_scheduler)
    print("Running first stage...")
    summary, schedule, order, mem_loc = scheduler.GetSolver(False)
    required_memory1 = summary["required_memory"]
    print("Required memory for task {}/ stage {}: {}".format(id, 1, required_memory1))

    print("Running second stage...")
    summary, schedule, order, mem_loc = scheduler.GetSolver(True, order)
    required_memory2 = summary["required_memory"]
    print("Required memory for task {}/ stage {}: {}".format(id, 2, required_memory2))

    if required_memory2 == 0:
        frag = 0
    else:
        frag = (required_memory2 - required_memory1) / required_memory2
    
    print("[+] Fragmentation for subtask {}: {}.".format(id, frag))

    # res = schedule
    visualizer.draw_schedule(schedule, img_path=f"./logs/graphs/{model_name}/{mode}/" + graph_name + "" + str(id) + ".png")
    res = dumps(schedule)
    schedules[id] = res
    return True