import os
import json


class Trace:
    def __init__(self, workload, trace):
        self.trace              = trace["workload_trace"]
        self.workload_info      = self.add_workload_info(workload)
        self.weighted_score     = (int(workload["weighted_score"]) if "weighted_score" in workload.keys() else 1)      # mJ to J
        self.fusable_kernels    = ["ReLU", "GeLU", "BatchNorm", "LayerNorm", "GroupNorm"] # "Softmax", "MaxPool2d", "AdaptiveAvgPool2d"

    def print_line(self):
        print("################################################")

    def print_trace_info(self):
        self.print_line()
        print("Workload Info")
        print(f"Model \t\t{self.workload_info['model']}")
        print(f"Model Instance\t{self.workload_info['model-instance']}")
        print(f"Batch Size\t{self.workload_info['batch-size']}")
        self.print_line()
    
    def get_model(self):
        return self.workload_info["model"]

    def get_model_instance(self):
        return self.workload_info["model-instance"]
    
    def get_batch_size(self):
        return self.workload_info["batch-size"]

    def add_workload_info(self, wl):
        wl_info = {
            "model": wl["model"],
            "model-instance": wl["model-instance"],
            "batch-size": wl["batch-size"]
        }
        return wl_info

    def get_num_kernels(self):
        return len(self.trace)

    def get_kernel(self, kernel_id=0):
        return self.trace[kernel_id] 
    
    def check_fuse_prev(self, kernel_id):
        # if first kernel, can't be fused with something previously
        if kernel_id == 0:
            return False

        # fuse any activations / batch norm / relu
        if self.trace[kernel_id]["kernel"] in self.fusable_kernels:
            return True
    
    def check_fuse_next(self, kernel_id):
        # if last kernel, can't be fused with something next
        if (kernel_id+1) == self.get_num_kernels():
            return False

        # fuse next activations / batch norm / relu
        if self.trace[kernel_id+1]["kernel"] in self.fusable_kernels:
            return True
        
    def get_fused_kernels(self):
        all_fused_kernels = []
        fused_kernels = []
        for kernel_id, kernel in enumerate(self.trace):
            fused_kernels.append(kernel)
            if not self.check_fuse_next(kernel_id):
                all_fused_kernels.append(fused_kernels)
                fused_kernels = []
        return all_fused_kernels 

class TraceParser:
    def __init__(self, trace_dir, experiment_file):
        self.all_traces         = []
        self.trace_dir          = trace_dir
        self.experiment_file    = experiment_file
        self.optimization_goal  = None
        self.allowed_opt_goals  = ["latency", "energy", "pareto"]
        
        self.extract_workloads()

    def add_trace(self, workload, trace_file):
        with open(trace_file, "r") as f:
            trace = json.load(f)
            t = Trace(workload, trace)
            self.all_traces.append(t)
        
    def find_trace_file(self, wl):
        likely_trace_file = self.trace_dir + "/" + wl["model"] + "/" + wl["model-instance"] + "-" + str(wl["batch-size"]) + "/" + wl["model"] + "-" + wl["model-instance"] + "-" + str(wl["batch-size"]) + "-bs.json"
        if not os.path.exists(likely_trace_file):
            print(f"Warning - trace file {likely_trace_file} could not be found...")
            exit(-1)
        return likely_trace_file

    def extract_workloads(self):
        if isinstance(self.experiment_file, str):
            with open(self.experiment_file, "r") as f:
                experiment = json.load(f)
        else:
            experiment = self.experiment_file

        for workload in experiment["workloads"]:
            tf = self.find_trace_file(workload)
            self.add_trace(workload, tf)
        
        if "optimization-goal" in experiment.keys():
            if experiment["optimization-goal"] not in self.allowed_opt_goals:
                print(f"Optimization goal not configured correctly ({self.allowed_opt_goals}) not in {experiment['optimization-goal']}")
                exit()
            self.optimization_goal = experiment["optimization-goal"]
        else:
            print("Warning - Optimization Goal not defined in experiment, setting to latency")
            self.optimization_goal = "latency"

    def get_trace(self, trace_id=0):
        return self.all_traces[trace_id]
    
    def get_traces(self):
        return self.all_traces
        
    




