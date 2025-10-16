import copy
import numpy as np

from scipy.optimize import LinearConstraint

class ShardKernel:
    def __init__(self):
        pass

    def shard_kernel(self, kernel, cut_dim, frac_work):
        sharding_strategy = None

        if cut_dim == "weights":
            # cut by weight
            if kernel["kernel"] == "Conv2d":
                kernel["output"]["out-channels"]    *= frac_work
                kernel["layer"]["out-channels"]     *= frac_work
                sharding_strategy = "weights"

            if kernel["kernel"] == "Linear":
                kernel["output"]["output-dim"]      *= frac_work
                kernel["layer"]["output-dim"]       *= frac_work
                sharding_strategy = "weights"

            if kernel["kernel"] == "EmbeddingTable":
                kernel["input"]["num-tables"]       *= frac_work
                kernel["output"]["num-tables"]      *= frac_work
                kernel["layer"]["num-tables"]       *= frac_work
                sharding_strategy = "data"
            
            if kernel["kernel"] == "InteractionArch":
                kernel["input"]["sharding_dim"]     = frac_work  # attention vals are broadcast but flops are fairly divided
                sharding_strategy = "weights"

            if kernel["kernel"] == "BatchNorm":
                kernel["input"]["in-channels"]      *= frac_work
                kernel["output"]["out-channels"]    *= frac_work
                sharding_strategy = "data"
            
            if kernel["kernel"] == "GroupNorm":
                kernel["input"]["in-channels"]      *= frac_work
                kernel["output"]["out-channels"]    *= frac_work
                sharding_strategy = "data"

            if kernel["kernel"] == "MaxPool2d":
                kernel["input"]["in-channels"]      *= frac_work
                kernel["output"]["out-channels"]    *= frac_work
                sharding_strategy = "data"
            
            if kernel["kernel"] == "AdaptiveAvgPool2d":
                kernel["input"]["in-channels"]      *= frac_work
                kernel["output"]["out-channels"]    *= frac_work
                sharding_strategy = "data"
        
        elif cut_dim == "batch":
            if kernel["kernel"] == "Conv2d":
                kernel["input"]["batch-size"]       *= frac_work
                kernel["output"]["batch-size"]      *= frac_work
                sharding_strategy = "data"

            if kernel["kernel"] == "Linear":
                kernel["input"]["batch-size"]       *= frac_work
                kernel["output"]["batch-size"]      *= frac_work
                sharding_strategy = "data"

            if kernel["kernel"] == "EmbeddingTable":
                kernel["input"]["batch-size"]       *= frac_work
                kernel["output"]["batch-size"]      *= frac_work
                sharding_strategy = "data"
            
            if kernel["kernel"] == "InteractionArch":
                kernel["input"]["batch-size"]       *= frac_work
                kernel["output"]["batch-size"]      *= frac_work  # attention vals are broadcast but flops are fairly divided
                kernel["input"]["sharding_dim"]     = 1
                sharding_strategy = "data"
            
            if kernel["kernel"] == "BatchNorm":
                kernel["input"]["batch-size"]       *= frac_work
                kernel["output"]["batch-size"]      *= frac_work
                sharding_strategy = "data"
            
            if kernel["kernel"] == "GroupNorm":
                kernel["input"]["batch-size"]       *= frac_work
                kernel["output"]["batch-size"]      *= frac_work
                sharding_strategy = "data"

            if kernel["kernel"] == "MaxPool2d":
                kernel["input"]["batch-size"]       *= frac_work
                kernel["output"]["batch-size"]      *= frac_work
                sharding_strategy = "data"
            
            if kernel["kernel"] == "AdaptiveAvgPool2d":
                kernel["input"]["batch-size"]       *= frac_work
                kernel["output"]["batch-size"]      *= frac_work
                sharding_strategy = "data"

        if kernel["kernel"] == "Attention":
            kernel["layer"]["num-heads"]            *= frac_work
            kernel["layer"]["output-dim"]           *= frac_work
            sharding_strategy = "weights"

        if kernel["kernel"] == "LayerNorm":
            kernel["input"]["seq-len"]              *= frac_work
            kernel["output"]["seq-len"]             *= frac_work
            kernel["input"]["beam-size"]            *= frac_work 
            kernel["output"]["beam-size"]           *= frac_work
            sharding_strategy = "data"
        

        if kernel["kernel"] == "GCNConv":
            kernel["input"]["num-edges"]        *= frac_work
            kernel["layer"]["sharding_dim"]     = frac_work
            kernel["output"]["num-edges"]       *= frac_work
            kernel["input"]["uncached"]         = True
            sharding_strategy = "data"

        # activations / fused kernels
        if kernel["kernel"] == "Add":
            kernel["input"]["elements"]         *= frac_work
            kernel["output"]["elements"]        *= frac_work
            sharding_strategy = "data"

        if kernel["kernel"] == "ReLU":
            kernel["input"]["elements"]         *= frac_work
            kernel["output"]["elements"]        *= frac_work
            sharding_strategy = "data"

        if kernel["kernel"] == "GeLU":
            kernel["input"]["elements"]         *= frac_work
            kernel["output"]["elements"]        *= frac_work
            sharding_strategy = "data"
        
        if kernel["kernel"] == "SiLU":
            kernel["input"]["elements"]         *= frac_work
            kernel["output"]["elements"]        *= frac_work
            sharding_strategy = "data"
        
        if kernel["kernel"] == "GEGLU":
            kernel["input"]["elements"]         *= frac_work
            kernel["output"]["elements"]        *= frac_work
            sharding_strategy = "data"
        
        if kernel["kernel"] == "Softmax":
            kernel["input"]["elements"]         *= frac_work
            kernel["output"]["elements"]        *= frac_work
            sharding_strategy = "data"
        

        if sharding_strategy == None:
            print("Warning: Sharding Strategy not implemented!")
            print(f"{kernel['kernel']} - {cut_dim}")
            exit()

        return kernel, sharding_strategy
    
    def shard_kernel_fused(self, fused_kernels, participating_chiplets, work_ratio, cut_dim="batch", verbose=0): # "batch", "weights"
        all_kernels = []
        for kernel in fused_kernels:
            kernel_shards = []
            for i, chiplet in enumerate(participating_chiplets):
                chiplet_name = chiplet.name
                chiplet_id   = chiplet.chiplet_id
                
                frac_work = work_ratio[i]

                shard, sharding_strategy = self.shard_kernel(copy.deepcopy(kernel), cut_dim, frac_work)
                kernel_shards.append([shard, sharding_strategy])
            all_kernels.append(kernel_shards)

        return all_kernels

    # create the final array for recording best work ratio
    # initalized with uniform distribution
    def init_best_work_ratio(self, num_fused_blocks, per_block_unique_participating_chiplets):
        self.best_work_ratio = [] 
        for b in range(num_fused_blocks):
            self.best_work_ratio += [1/per_block_unique_participating_chiplets[b]] * per_block_unique_participating_chiplets[b] # put it in the middle for now
    
    # return work ratios with zeros for each chiplet group
    def get_all_init_work_distributions(self, participating_chiplets, dse_complexity):
        unique_chiplet_names, chiplet_instances = self.get_unique_chiplet_types(participating_chiplets)
        init_work_dist = []
        for i, chiplet_name in enumerate(unique_chiplet_names):
            this_chiplet_work_dist = []
            if dse_complexity == "individual":
                for c in participating_chiplets:
                    frac_work = (1.0/chiplet_instances[i]) if (c.name == chiplet_name) else 0.0
                    this_chiplet_work_dist.append(frac_work)
            else:
                for c in unique_chiplet_names:
                    if c == chiplet_name:
                        this_chiplet_work_dist.append(1.0)
                    else:
                        this_chiplet_work_dist.append(0.0)
            init_work_dist.append(this_chiplet_work_dist)
        
        # add a random init point
        if dse_complexity == "individual":
            rand_vals = np.random.rand(len(participating_chiplets))
            init_work_dist.append(rand_vals / sum(rand_vals)) # normalize to 1
        else:
            rand_vals = np.random.rand(len(unique_chiplet_names))
            init_work_dist.append(rand_vals / sum(rand_vals)) # normalize to 1

        return init_work_dist

    def get_unique_chiplet_types(self, participating_chiplets):
        chiplet_names = []
        chiplet_instances = []
        for chiplet in participating_chiplets:
            if chiplet.name not in chiplet_names:
                chiplet_names.append(chiplet.name)
                chiplet_instances.append(0)
            chiplet_instances[chiplet_names.index(chiplet.name)] += 1
        return chiplet_names, chiplet_instances

    def init_shard_dse_grouped(self, participating_chiplets):
        self.unique_chiplet_names, self.chiplet_instances = self.get_unique_chiplet_types(participating_chiplets)
        self.bounds = [(0.0, 1.0)] * len(self.unique_chiplet_names) # init bounds
        self.constraints = [LinearConstraint(np.ones(len(self.unique_chiplet_names)), 1.0, 1.0)]  # init constraints - must sum to 1
    
    def init_shard_dse_individual(self, participating_chiplets):
        num_participating_chiplets = len(participating_chiplets)
        self.bounds = [(0.0, 1.0)] * num_participating_chiplets # init bounds
        self.constraints = [LinearConstraint(np.ones(num_participating_chiplets), 1.0, 1.0)] # init constraints - must sum to 1
    
    # create the linear constraints for each knob of work distribution
    def init_shard_dse(self, participating_chiplets, dse_complexity):
        if dse_complexity == "grouped":
            self.init_shard_dse_grouped(participating_chiplets)
        elif dse_complexity == "individual": 
            self.init_shard_dse_individual(participating_chiplets)
        else:
            print("Unkown value for dse_complexity:", dse_complexity)
            exit()

    # used for taking a work ratio and if it is grouped, expanded so that all chiplets have a fraction of work
    # minimize will either offer a grouped or individual work distribution, this returns the breakdown per chiplet
    def expand(self, work_ratio, participating_chiplets, dse_complexity):
        if dse_complexity == "grouped":
            unique_chiplet_names, chiplet_instances = self.get_unique_chiplet_types(participating_chiplets)
            total_workers = np.dot(np.array(work_ratio), np.array(chiplet_instances))
            
            expanded_work_ratio = []
            for chiplet in participating_chiplets:
                chiplet_index = unique_chiplet_names.index(chiplet.name)
                frac_work = work_ratio[chiplet_index] / total_workers
                expanded_work_ratio.append(frac_work)
            return expanded_work_ratio 
        elif dse_complexity == "individual": 
            return np.array(work_ratio) 

            # uniform work partitioning
            #expanded_work_ratio = [1/len(participating_chiplets)] * len(participating_chiplets)
            
            # flop based partitioning 
            #if len(participating_chiplets) > 1:
            #    expanded_work_ratio = [.18, .272, .272, .272]
            
    

