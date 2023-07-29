import sys
import copy

def test():
    pass

class Subtask:
    def __init__(
        self,
        graph,
        id,
        fwd,
        inner_fwd,
        inner_bwd,
        bwd,
        spans,
        solver_spans,
        all_activations
    ):       
        self.graph = graph
        self.id = id
        self.fwd = fwd
        self.inner_fwd = inner_fwd
        self.inner_bwd = inner_bwd
        self.bwd = bwd
        self.spans = spans
        self.solver_spans = solver_spans
        self.re_spans = solver_spans \
                    if solver_spans else spans
        self.all_activations = all_activations
        self.full_live_full_add = set()
        self.sub_live_full_add = set()
        self.sub_live_no_add = set()
        self.nodes = set() 
        self.activations = set()
        # CIFI: 0
        # CIFO: 1
        # COFI: 2
        # bound_tensors: 3
        self.tensor_attribute = dict()

    def is_in_gap(self, num, section):
        if section[0] <= num and num <= section[1]: 
            return True
        else:
            return False

    def is_tensor_inner(self, bound, mul):
        bound1, bound2 = bound[0], bound[1]
        bound3, bound4 = bound[2], bound[3]
        lb, ub = mul[0], mul[1]
        
        is_create = False
        is_free = False

        if self.is_in_gap(lb, [bound1, bound2]) or \
            self.is_in_gap(lb, [bound3, bound4]):
            is_create = True

        if bound2 == bound3 and lb == bound2:
            is_create = True
        
        if self.is_in_gap(ub, [bound1, bound2]) or \
            self.is_in_gap(ub, [bound3, bound4]):
            is_free = True
        
        if is_create and is_free:
            return "CIFI"
        elif is_create and not is_free:
            return "CIFO"
        elif not is_create and is_free:
            return "COFI"

    def get_task_info(self):
        asap = self.spans[0]
        alap = self.spans[1]
        MUL = self.spans[2]
        
        bound1 = -999999 if not self.fwd else asap[self.fwd]
        bound2, bound3 = asap[self.inner_fwd], asap[self.inner_bwd]
        bound4 = 999999 if not self.bwd else asap[self.bwd]
        bound = [bound1, bound2, bound3, bound4]

        BFS = [self.inner_fwd]    
        while len(BFS) != 0:
            node = BFS.pop(0)
            self.nodes.add(node)

            for e in node.fanin:
                if node is self.fwd:
                    continue
                source = e.source
                if asap[source] >= bound1 and source not in self.nodes and source not in BFS:
                    BFS.append(e.source)
                
                rel = self.is_tensor_inner(bound, MUL[e])
                if rel == "CIFI":
                    self.full_live_full_add.add(e)
                    self.tensor_attribute[e] = 0
                    if e in self.all_activations:
                        self.activations.add(e)   
                elif rel == "COFI":
                    # import pdb;pdb.set_trace()
                    self.sub_live_full_add.add(e)
                    self.tensor_attribute[e] = 2
                    if e in self.all_activations:
                        self.activations.add(e)
            
            for e in node.fanout:
                if node is self.inner_fwd:
                    continue
                rel = self.is_tensor_inner(bound, MUL[e])
                if rel == "CIFI":
                    self.tensor_attribute[e] = 0
                    self.full_live_full_add.add(e)
                elif rel == "CIFO":
                    self.tensor_attribute[e] = 1
                    self.sub_live_no_add.add(e)
                
        BFS = [self.inner_bwd]
        while len(BFS) != 0:
            node = BFS.pop(0)
            self.nodes.add(node)
            
            for e in node.fanout:
                if node is self.bwd:
                    continue
                
                rel = self.is_tensor_inner(bound, MUL[e])
                for sink in e.sinks:
                    if alap[sink] <= bound4 and sink not in self.nodes and sink not in BFS:
                        BFS.append(sink)

                if rel == "CIFI":
                    self.tensor_attribute[e] = 0
                    self.full_live_full_add.add(e)
                    if e in self.all_activations:
                        self.activations.add(e)
                
                elif rel == "CIFO":
                    self.tensor_attribute[e] = 1 
                    self.sub_live_full_add.add(e)  
                    if e in self.all_activations:
                        self.activations.add(e)   

            for e in node.fanin:
                if node is self.inner_bwd:
                    continue
                source = e.source
                rel = self.is_tensor_inner(bound, MUL[e])
                if rel == "CIFI":
                    self.tensor_attribute[e] = 0
                    self.full_live_full_add.add(e)
                    if e in self.all_activations:
                        self.activations.add(e)
                elif rel == "COFI":
                    self.tensor_attribute[e] = 2
                    if e in self.all_activations:
                        self.sub_live_full_add.add(e)
                        self.activations.add(e)
                    else:
                        self.sub_live_no_add.add(e)

        if self.inner_fwd and self.inner_fwd.name != "sum_1":
            for e in self.inner_fwd.fanout:
                self.tensor_attribute[e] = 3
                self.sub_live_no_add.add(e)
       
        if self.bwd:
            for e in self.bwd.fanout:
                self.tensor_attribute[e] = 3
                self.sub_live_no_add.add(e)
        return self.full_live_full_add, self.sub_live_full_add, self.sub_live_no_add, self.activations, self.nodes           

    def get_tensors_nodes(self):
        return self.get_task_info()

    def get_info(self):
        return self.graph, [self.fwd, self.inner_fwd], [self.bwd, self.inner_bwd], self.re_spans