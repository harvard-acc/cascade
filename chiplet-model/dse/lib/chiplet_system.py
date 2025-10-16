import os
import time
import json
import glob
from tqdm import tqdm

import statistics
import numpy as np
from scipy.optimize import minimize

from lib.shard_kernel import ShardKernel

class ChipletSystem:
    def __init__(self, chiplet_library_dir, verbose=1):
        self.chiplet_library_dir    = chiplet_library_dir
        self.verbose                = verbose

        self.chiplet_library        = self.register_chiplets()
        self.sk                     = ShardKernel()

        self.mem_bw                 = None          # GB/sec
        self.mem_bw_g               = None          # B/sec
        self.optimization_goal      = None
        self.current_block_dse      = -1

        self.POWER_PER_CHANNEL      = 0.5               # Watts / DRAM channel on-power
        self.PJ_PER_BIT             = 4.5*(10**-(12))    # 4.5 pJ/bit
        
        self.first_energy           = []

        if self.verbose > 0:
            self.print_chiplet_info()

    def print_line(self):
        print("################################################")

    def print_chiplet_info(self):
        self.print_line()
        print("Chiplet Library")
        for i, chiplet_name in enumerate(self.chiplet_library):
            print()
            self.chiplet_library[chiplet_name]().print_info()
        self.print_line()
    
    def print_chiplet_system_info(self):
        self.print_line()
        print("Chiplet System")
        for i, chiplet in enumerate(self.chiplet_soc):
            print()
            chiplet.print_info()
        self.print_line()

    def register_chiplets(self):
        import sys
        sys.path.append(self.chiplet_library_dir)

        from register_chiplet import RegisterChiplets
        chiplet_library = RegisterChiplets().register_chiplets()

        return chiplet_library

    def reset_chiplet_cache(self):
        for chiplet in self.chiplet_library:
            self.chiplet_library[chiplet].cached_data = 0

    def init_system_bandwidth(self, bw_per_channel, num_channels):
        self.num_channels = num_channels
        self.mem_bw = bw_per_channel * num_channels
        self.mem_bw_g = bw_per_channel * num_channels * (1024**3)
        #for chiplet in self.chiplet_soc:
        #    chiplet.set_interconnect_bw(chiplet_bw)
        
    def init_warmup(self):
        for chiplet in self.chiplet_soc:
            chiplet.init_warmup()
    
    def init_eval(self):
        for chiplet in self.chiplet_soc:
            chiplet.init_eval()

    def configure_system(self, desired_chiplets, optimization_goal):
        self.chiplet_soc = []
        for i, chiplet in enumerate(desired_chiplets):
            self.chiplet_soc.append(self.chiplet_library[chiplet](i))
        
        if self.verbose > 0:
            self.print_chiplet_system_info()
        self.optimization_goal = optimization_goal
    
    def get_num_chiplets(self):
        return len(self.chiplet_soc)

    def compute_roofline(self):
        peak_bf16_flops = 0
        for chiplet in self.chiplet_soc:
            peak_bf16_flops += chiplet.bf16_flops
        return peak_bf16_flops

    def bandwidth_roofline(self):
        peak_bw_sat = 0
        for chiplet in self.chiplet_soc:
            peak_bw_sat += chiplet.bw_sat
        return min(self.mem_bw, peak_bw_sat)
    
    def checkpoint_chiplets(self):
        for chiplet in self.chiplet_soc:
            chiplet.checkpoint()
    
    def replay_chiplets(self, participating_chiplets):
        for chiplet in participating_chiplets:
            chiplet.replay()

     # only chipelts which can execute the kernels in the region can participate!
    # participating chiplets do not change during forward fused blocks
    # clear the cached_data if some chiplets go from participating to not participating
    def get_participating_chiplets(self, fused_kernel_block):
        participating_chiplets = []
        for chiplet in self.chiplet_soc:
            chiplet_can_participate = True
            for kernel in fused_kernel_block:
                if (kernel["kernel"] not in chiplet.kernels) and (not "all" in chiplet.kernels):
                    chiplet_can_participate = False
            if chiplet_can_participate:
                participating_chiplets.append(chiplet)
            else:
                chiplet.clear_cache()
        if len(participating_chiplets) == 0:
            if self.verbose > 0:
                print("No chiplets are able to participate in this kernel - must invalidate this chiplet_soc configuration!")
            return -1
        return participating_chiplets
    
    def get_chiplet_ids(self, participating_chiplets):
        chiplet_ids = []
        for chiplet in participating_chiplets:
            chiplet_ids.append(chiplet.chiplet_id)
        return chiplet_ids

    def record_data_init(self, kernel_name, chiplet_ids):
        self.kernel_perf    = {"name": kernel_name, "chiplets": {}, "total": {"flops": 0, "mem_accessed": 0, "exe_time": 0, "energy": 0, "energy_dram": 0}}
        for c in chiplet_ids:
            self.kernel_perf["chiplets"][c] = {"flops": 0, "mem_accessed": 0, "exe_time": 0, "energy": 0, "work": 0}

        num_chiplets        = len(self.chiplet_soc)
        self.total_flops    = np.zeros(num_chiplets)
        self.private_data   = np.zeros(num_chiplets)
        self.shared_data    = np.zeros(num_chiplets)
        self.all_exe_time   = np.zeros(num_chiplets)
        self.all_energy     = np.zeros(num_chiplets)
        
    def record_analytical_data(self, chiplet_id, analytical_perf, sharding_strategy):
        # record data
        self.total_flops[chiplet_id]        = analytical_perf[0]

        if sharding_strategy == "weights":
            self.shared_data[chiplet_id]    = analytical_perf[1]["data"]
            self.private_data[chiplet_id]   = analytical_perf[1]["weight"] + analytical_perf[1]["out"]
        else:
            self.shared_data[chiplet_id]    = analytical_perf[1]["weight"]
            self.private_data[chiplet_id]   = analytical_perf[1]["data"] + analytical_perf[1]["out"]
        
        # chiplet data        
        self.kernel_perf["chiplets"][chiplet_id]["flops"]           = self.total_flops[chiplet_id]
        self.kernel_perf["chiplets"][chiplet_id]["mem_accessed"]    = self.shared_data[chiplet_id] + self.private_data[chiplet_id]
    
    def record_execution_data(self, chiplet_id, chiplet, perf, work):
        exe_time                                                    = perf["exe_time"]
        energy                                                      = perf["energy"]
        self.all_exe_time[chiplet_id]                               = exe_time
        self.all_energy[chiplet_id]                                 = energy

        self.kernel_perf["chiplets"][chiplet_id]["name"]            = chiplet.name
        self.kernel_perf["chiplets"][chiplet_id]["exe_time"]        = exe_time                  # seconds
        self.kernel_perf["chiplets"][chiplet_id]["energy"]          = energy                    # energy in J
        self.kernel_perf["chiplets"][chiplet_id]["work"]            = work                      # fraction of work for kernel
        
    def record_total_analytical_data(self):
        self.kernel_perf["total"]["flops"]                          = np.sum(self.total_flops)
        self.kernel_perf["total"]["mem_accessed"]                   = np.max(self.shared_data) + np.sum(self.private_data)        
    
    def record_total_perf_data(self):
        self.kernel_perf["total"]["exe_time"]                       = np.max(self.all_exe_time)    # dictated by slowest chiplet
        self.kernel_perf["total"]["energy"]                         = np.sum(self.all_energy)
        self.kernel_perf["total"]["energy_dram"]                    = self.kernel_perf["total"]["exe_time"] * self.num_channels * self.POWER_PER_CHANNEL + self.kernel_perf["total"]["mem_accessed"] * 8 * self.PJ_PER_BIT
    
    def clean_work_distribution(self, work_distribution, cuttoff=0.05):
        arr = np.array(work_distribution)
        arr = np.where((arr < (cuttoff/100)), 0, arr)
        total = np.sum(arr)
        return arr/total

    def characterize_kernel(self, participating_chiplets, kernel_shards, work_distribution, fuse_prev, fuse_next):
        #########################################################################
        # analytical - flops and mem accessed  -  10-20 us
        # send shard to each chiplet to profile
        self.record_data_init(kernel_shards[0][0]["kernel"], self.get_chiplet_ids(participating_chiplets))

        for chiplet_id, chiplet in enumerate(participating_chiplets):
            # characterize kernel flops and mem accesses using level-1 / 3 
            shard, sharding_strategy = kernel_shards[chiplet_id]
            analytical_perf = chiplet.get_analytical(shard, fuse_prev, fuse_next, self.dtype, self.verbose)     # 5-10 us
            self.record_analytical_data(chiplet.chiplet_id, analytical_perf, sharding_strategy)                         # <1 us
        #########################################################################
        self.record_total_analytical_data()

        #########################################################################
        # calc execution time and energy  -  5-6 us
        # get the exe_time using level-1 roofline /level-3 trace
        all_exe_time = []
        for chiplet_id, chiplet in enumerate(participating_chiplets):
            # calculate the chiplet memory bandwidth - assume all chiplets have same bandwidth priority (governed by volume of requested data)!
            chiplet_data = self.kernel_perf["chiplets"][chiplet.chiplet_id]["mem_accessed"] # not sure when chiplet_data will be zero and work_distribution isn't 0... need to consider
            if chiplet_data == 0 or work_distribution[chiplet_id] == 0:
                perf = {"exe_time": 0, "energy": 0}
            else:
                all_data = self.kernel_perf["total"]["mem_accessed"]
                chiplet_mem_bw = chiplet_data / all_data * self.mem_bw_g
                perf = chiplet.get_perf(chiplet_data, chiplet_mem_bw, self.verbose)                         # ~ 2us
            self.record_execution_data(chiplet.chiplet_id, chiplet, perf, work_distribution[chiplet_id])    # <.5us
        self.record_total_perf_data()

        #########################################################################

        return self.kernel_perf
    

    def characterize_fused_block(self, base_block_id, fused_kernel_block, participating_chiplets, work_distribution):
        # create shards from optimized work sharding
        fused_kernel_shards = self.sk.shard_kernel_fused(fused_kernel_block, participating_chiplets, work_distribution, cut_dim=self.cut_dim, verbose=self.verbose)
        kernel_results = []
        for kernel_id, kernel_shards in enumerate(fused_kernel_shards):
            if self.verbose:
                print("#################################")
                print(f"{kernel_shards[0][0]['kernel']} - Layer {base_block_id + kernel_id}")
                
            fuse_prev = (kernel_id != 0)
            fuse_next = (kernel_id+1 != len(fused_kernel_block))

            # desired metrics per kernel: latency, flops, mem_accesses
            perf = self.characterize_kernel(participating_chiplets, kernel_shards, work_distribution, fuse_prev, fuse_next) # 33us ish
            kernel_results.append(perf)

        if self.verbose > 0:
            print()
            for chiplet_id, chiplet in enumerate(participating_chiplets):
                chiplet_exe_time = 0
                chiplet_energy = 0
                for perf in kernel_results:
                    chiplet_exe_time += perf["chiplets"][chiplet.chiplet_id]["exe_time"]
                    chiplet_energy += perf["chiplets"][chiplet.chiplet_id]["energy"]
                print(f"\tFused Perf - {chiplet.name.capitalize()} Chiplet: \t%0.2f%% - %0.2fus - %0.2fuJ" % ((work_distribution[chiplet_id]*100), chiplet_exe_time*(10**6), chiplet_energy*(10**6)))
            print("\tWork Distribution: [", end="")
            for w in work_distribution:
                print(" %0.2f%%" % (w*100), end="")
            print(" ]\n")
        return kernel_results


    def get_best_block_work_dist(self, fused_block_id, participating_chiplets):
        start, end = sum(self.per_block_participating_chiplets[:fused_block_id]), sum(self.per_block_participating_chiplets[:fused_block_id+1])
        return self.sk.expand(self.sk.best_work_ratio[start:end], participating_chiplets, self.dse_complexity)
    
    def get_best_block_work_dist_not_expanded(self, fused_block_id, participating_chiplets):
        start, end = sum(self.per_block_participating_chiplets[:fused_block_id]), sum(self.per_block_participating_chiplets[:fused_block_id+1])
        return self.sk.best_work_ratio[start:end]

    def set_block_work_dist(self, fused_block_id, work_distribution):
        start, end = sum(self.per_block_participating_chiplets[:fused_block_id]), sum(self.per_block_participating_chiplets[:fused_block_id+1])
        self.sk.best_work_ratio[start:end] = work_distribution

    def score(self, perf, participating_chiplets, work_distribution, optimization_goal="latency", verbose=False):
        fused_kernel_latency    = [0] * len(participating_chiplets)
        fused_kernel_energy     = [0] * len(participating_chiplets)
        for kernel in perf:
            worst_latency = 0
            for i, chiplet_id in enumerate(kernel["chiplets"]):
                chiplet_latency = kernel["chiplets"][chiplet_id]["exe_time"]
                chiplet_energy  = kernel["chiplets"][chiplet_id]["energy"]
                worst_latency   = max(chiplet_latency, worst_latency)

                fused_kernel_latency[i] += chiplet_latency
                fused_kernel_energy[i]  += chiplet_energy  
            
            # if including DRAM power in energy optimization. Probably shouldn't because DRAM will in theory scale with the BW saturation of each chiplet
            #if optimization_goal == "energy":
            #    fused_kernel_metric[0] += worst_latency * self.POWER_PER_CHANNEL * self.num_channels

        # calculate final performance metric for a fused block
        if optimization_goal == "latency":
            if len(participating_chiplets) > 1:
                model_dse_metric = statistics.variance(np.array(fused_kernel_latency)/sum(fused_kernel_latency)) * (10**6)
                #if verbose:
                #    print(perf[0]["name"], model_dse_metric, work_distribution, np.array(fused_kernel_latency))
                #print(perf[0]["name"], model_dse_metric, sum(work_distribution), "[{}]".format(", ".join("{:.3f}".format(num) for num in work_distribution)), "[{}]".format(", ".join("{:.3f}".format(num*1000) for num in fused_kernel_latency)))
            else:
                model_dse_metric = 0 # dse has no impact because only 1 cut
        if optimization_goal == "energy":
            energy_sum = np.sum(fused_kernel_energy)
            model_dse_metric = energy_sum * (10**6)
            #print(perf[0]["name"], model_dse_metric, sum(work_distribution), work_distribution)
        
        return model_dse_metric


    def characterize_workload_dse(self, STOP_BLOCK): 
        np.random.seed(42)
        
        # warmup system based on chosen 
        self.init_warmup() 
        base_block_id = 0
        for fused_block_id, fused_kernel_block in enumerate(self.all_fused_kernels):
            # get the participating chiplets
            participating_chiplets = self.get_participating_chiplets(fused_kernel_block)

            # returns the current known best work distribution 
            block_work_dist = self.get_best_block_work_dist(fused_block_id, participating_chiplets)
            self.characterize_fused_block(base_block_id, fused_kernel_block, participating_chiplets, block_work_dist)
            base_block_id += len(fused_kernel_block)

            if fused_block_id == STOP_BLOCK:
                break
            
        # eval
        # fast forward to kernel being minimized and checkpoint then replay until minimization is done
        self.init_eval() 
        base_block_id = 0
        for fused_block_id, fused_kernel_block in enumerate(self.all_fused_kernels):
            # checkpoint the chiplets - save cached data and available cache etc.
            self.checkpoint_chiplets()
            
            # get participating chiplets in problem
            participating_chiplets = self.get_participating_chiplets(fused_kernel_block)     

            # get the work distribution per block - list of all possible starting points
            all_init_work_distr = [self.get_best_block_work_dist_not_expanded(fused_block_id, participating_chiplets)] + self.sk.get_all_init_work_distributions(participating_chiplets, self.dse_complexity)

            current_block_score = None
            current_block_best_score = np.inf

            # for each initial point to try
            for init_work_distribution in all_init_work_distr:
                # create Linear Contraints for each chiplet (either grouped or individual)
                self.sk.init_shard_dse(participating_chiplets, self.dse_complexity)

                def minimize_fused_block(work_distribution):
                    self.replay_chiplets(participating_chiplets)
                    work_distribution = self.sk.expand(work_distribution, participating_chiplets, self.dse_complexity)
                    perf = self.characterize_fused_block(base_block_id, fused_kernel_block, participating_chiplets, work_distribution)
                    return self.score(perf, participating_chiplets, work_distribution, optimization_goal=self.optimization_goal) # "pareto"
                
                result = minimize(minimize_fused_block, init_work_distribution, bounds=self.sk.bounds, constraints=self.sk.constraints) # , options={"disp": True} , options={"eps": .1}, "ftol": .00000000001
                
                # get final performance and set the best work distribution
                self.replay_chiplets(participating_chiplets)
                work_distribution = self.sk.expand(result.x, participating_chiplets, self.dse_complexity)
                perf = self.characterize_fused_block(base_block_id, fused_kernel_block, participating_chiplets, work_distribution)
                current_block_score = self.score(perf, participating_chiplets, work_distribution, optimization_goal=self.optimization_goal) 

                if current_block_score < current_block_best_score:
                    current_block_best_score = current_block_score
                    self.set_block_work_dist(fused_block_id, result.x) # set the new best work distribution 

            # run the best work distribution
            self.replay_chiplets(participating_chiplets)
            work_distribution = self.get_best_block_work_dist(fused_block_id, participating_chiplets)
            perf = self.characterize_fused_block(base_block_id, fused_kernel_block, participating_chiplets, work_distribution)
            current_block_score = self.score(perf, participating_chiplets, work_distribution, optimization_goal=self.optimization_goal, verbose=True)
            
            # minimization function for finding best work distribution
            base_block_id += len(fused_kernel_block)
            if fused_block_id == STOP_BLOCK:
                break


    def characterize_workload_infer(self, stop_block=-1): 
        self.verbose, old_verbose = 0, self.verbose
        self.init_warmup() 
        base_block_id = 0
        for fused_block_id, fused_kernel_block in enumerate(self.all_fused_kernels):
            participating_chiplets = self.get_participating_chiplets(fused_kernel_block)
            block_work_dist = self.clean_work_distribution(self.get_best_block_work_dist(fused_block_id, participating_chiplets))
            self.characterize_fused_block(base_block_id, fused_kernel_block, participating_chiplets, block_work_dist)
            base_block_id += len(fused_kernel_block)
            if fused_block_id == stop_block:
                break
        self.verbose = old_verbose

        # eval
        self.init_eval() 
        kernel_results = []
        base_block_id = 0
        for fused_block_id, fused_kernel_block in enumerate(self.all_fused_kernels): 
            participating_chiplets = self.get_participating_chiplets(fused_kernel_block)            
            block_work_dist = self.clean_work_distribution(self.get_best_block_work_dist(fused_block_id, participating_chiplets))
            perf = self.characterize_fused_block(base_block_id, fused_kernel_block, participating_chiplets, block_work_dist)
            kernel_results += perf
            base_block_id += len(fused_kernel_block)
            if fused_block_id == stop_block:
                break
        return kernel_results


    def get_per_block_participating_chiplets(self, dse_complexity):
        per_block_participating_chiplets = []
        for fused_kernel_block in self.all_fused_kernels:
            pc = self.get_participating_chiplets(fused_kernel_block)
            if dse_complexity == "grouped":
                per_block_participating_chiplets += [len(self.sk.get_unique_chiplet_types(pc)[0])]
            elif dse_complexity == "individual":
                per_block_participating_chiplets += [len(pc)]
            else:
                print("Unknown dse_complexity:", dse_complexity)
                exit()
        return per_block_participating_chiplets


    def check_valid_system(self, trace):
        self.all_fused_kernels = trace.get_fused_kernels()
        for fused_kernel_block in self.all_fused_kernels:
            pc = self.get_participating_chiplets(fused_kernel_block)
            if pc == -1:
                return -1
            for c in pc:
                if c.name in self.chiplet_validate:
                    self.chiplet_validate.pop(self.chiplet_validate.index(c.name))
        return 0


    def characterize_workload(self, trace, cut_dim="weights", dtype=2):
        self.dtype = dtype
        if self.verbose > 0:
            trace.print_trace_info()


        STOP_BLOCK = -1
        self.cut_dim = cut_dim # "weights" "batch" - how to shard the model per kernel - will likely need a knob as well

        self.dse_complexity = "grouped"     # "grouped" "individual" - are types of chiplets grouped together, or does each chiplet operate independently? grouped means chiplets of the same group have identical work distribution, individual means they are independent from each other which inceases completxity but may yield higher performance in some cases

        self.verbose, old_verbose = 0, self.verbose
        self.all_fused_kernels = trace.get_fused_kernels()
        self.per_block_participating_chiplets = self.get_per_block_participating_chiplets(self.dse_complexity)
        
        self.sk.init_best_work_ratio(len(self.all_fused_kernels), self.per_block_participating_chiplets)

        # may need to do multiple passes because what is cached will change with work distribution
        # iterate over all_fused_kernels and minimize the balance the local kernel execution time
        t1 = time.perf_counter_ns()
        for i in range(3):
            self.characterize_workload_dse(STOP_BLOCK) 

        t2 = time.perf_counter_ns()
        self.verbose = old_verbose
        
        # Do a final characterization to show that the kernel's times are almost all the same
        kernel_results = self.characterize_workload_infer(STOP_BLOCK)
        
        if self.verbose > 0:
            print("DSE Time: %0.2fs" % ((t2-t1)/(10**9)))

        return kernel_results


