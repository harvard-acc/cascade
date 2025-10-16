import os
import time
import math
import random 

import numpy as np

class BaseChiplet:
    def __init__(self, chiplet_id=None):
        self.name               = ""
        self.chiplet_id         = chiplet_id
        self.bf16_flops         = 0
        self.bw_sat             = 0
        self.tdp                = 0
        self.area               = 0
        self.sram               = 0
        self.kernels            = []
        self.cached_leak        = 3/100         # fraction of data leaked from on-chip to memory during kernel fusion
        self.cache_buf_scalar   = 1.2           # fraction noise of leaked cache data

        self.chiplet_bw         = None          # GB/sec

        self.cached_data        = 0
        self.max_cached_data    = 0
        self.warmup             = False
        self.available_cache    = 0             # default says that there's no cache available for saving weights during forward pass

        self.bf16_flops_g       = 0
        self.bw_sat_g           = 0
        self.chiplet_bw_g       = 0
        self.min_chiplet_bw_g   = None
        self.database           = {}            # used for saving previously executed runs

        self.kernel_mapping = {
            "Linear":                   self.linear,
            "Conv2d":                   self.conv_2d,
            "Add":                      self.add,
            "EmbeddingTable":           self.embedding_bag_collection,
            "InteractionArch":          self.interaction_arch,
            "Attention":                self.attention,
            "GCNConv":                  self.gcn_conv,
            "ReLU":                     self.relu,
            "GeLU":                     self.gelu,
            "SiLU":                     self.silu,
            "Softmax":                  self.softmax,
            "BatchNorm":                self.batch_norm,
            "GroupNorm":                self.group_norm,
            "LayerNorm":                self.layer_norm,
            "MaxPool2d":                self.max_pool_2d,
            "AdaptiveAvgPool2d":        self.adaptive_avg_pool_2d,
            # Add mappings for other layers as needed
        }
    
    def finish_init(self):
        self.bf16_flops_g = self.bf16_flops * (10**9)
        self.bw_sat_g = self.bw_sat * (1024**3)
        self.sram_g = self.sram * (1024**2)
        self.tdp_compute = .50 * self.tdp               # compute power following load line
        self.tdp_mem_controllers = .30 * self.tdp       # mem controllers and mesh following peak bandwidth
        self.tdp_base = .20 * self.tdp                  # base on power which must be paid regardless
        self.min_chiplet_bw_g = self.bw_sat_g

    #def set_interconnect_bw(self, bw):
    #    self.chiplet_bw         = bw
    #    self.chiplet_bw_g       = bw * (1024**3)
    #    self.min_chiplet_bw_g   = min(self.chiplet_bw_g, self.bw_sat_g)
    
    def get_flops(self, kernel_name):
        return self.bf16_flops_g

    def get_on_chip_capacity(self):
        return self.sram_g
    
    def init_warmup(self):
        self.warmup = True
        self.max_cached_data = 0

    def init_eval(self):
        self.warmup = False
        self.available_cache = max(self.get_on_chip_capacity() - self.max_cached_data, 0)
        #print("Available Cache: %0.2f MB" % (self.available_cache/(1024**2)))

    def clear_cache(self):
        self.cached_data = 0
    
    def checkpoint(self):
        self.saved_max_cached_data = self.max_cached_data
        self.saved_cache = self.cached_data
        self.saved_available_cache = self.available_cache
    
    def replay(self):
        self.max_cached_data = self.saved_max_cached_data
        self.cached_data = self.saved_cache
        self.available_cache = self.saved_available_cache
        
    def print_info(self):
        if self.chiplet_id is not None:
            print("ID: \t\t", self.chiplet_id)
        print("Name: \t\t", self.name)
        print("BF16 FLOPS: \t", self.bf16_flops, "\tGFLOPs")
        print("TDP: \t\t", self.tdp, "\tW")
        print("Area: \t\t", self.area, "\tmm^2")
        print("Kernels: \t", ("\t" + str(self.kernels)))
        print("BW Sat: \t", self.bw_sat, "\tGB/sec")
        #print("Chiplet BW: \t", self.chiplet_bw, "\tGB/sec")
    
    def dtype_aware(self, perf, dtype):
        perf[1]["data"]     *= dtype
        perf[1]["weight"]   *= dtype
        perf[1]["out"]      *= dtype
    
    def print_analytical_perf(self, perf):
        print("\tCompute: \t\t%0.5f" % (perf[0]/(10**6)), "MFLOPs")
        print("\tData: \t\t\t%0.5f" % (perf[1]["data"]/(1024**2)), "MB")
        print("\tWeights: \t\t%0.5f" % (perf[1]["weight"]/(1024**2)), "MB")
        print("\tOutput: \t\t%0.5f" % (perf[1]["out"]/(1024**2)), "MB")
        print("\tTotal: \t\t\t%0.5f" % ((perf[1]["data"] + perf[1]["weight"] + perf[1]["out"])/(1024**2)), "MB")

    def get_analytical_kernel(self, kernel_shard):
        kernel_name = kernel_shard["kernel"]
        kernel_analytical = self.kernel_mapping.get(kernel_name)

        if not kernel_analytical:
            print(f"Warning: Layer not implemented {kernel_name}\n")
            return 

        return kernel_analytical

    def get_analytical(self, kernel_shard, fuse_prev, fuse_next, dtype, verbose=0):
        # chiplets can override get_analytical kernel to integrate level 3 or more accurate level 1 kernel
        self.dtype = dtype

        # get kernel
        kernel_analytical_model = self.get_analytical_kernel(kernel_shard)
        if not kernel_analytical_model:
            return 
                
        perf = kernel_analytical_model(kernel_shard)
        self.dtype_aware(perf, dtype)
        total_kernel_data = perf[1]["data"] + perf[1]["weight"] + perf[1]["out"]

        # check if kernel input has a hint that the data is uncached
        if "uncached" in kernel_shard["input"].keys():
            self.cached_data = 0

        # if the current kernel is fused with the previous kernel
        # then the data should be used as generated and only the "leaked" data should be read
        # if not fused, then you can read some of the data from your cache 
        if fuse_prev:
            perf[1]["data"] *= (self.cached_leak)
        else:
            all_data_in = perf[1]["data"]
            perf_data = (all_data_in - self.cached_data) + (all_data_in * self.cached_leak)
            perf[1]["data"] = perf_data

            if perf[1]["data"] < 0:
                # This warning is no longer a warning - but good to have as a sanity check in future because when only 2 chiplets participate, then 4 chiplets, the 2 previously participating hold all data and cause this scenario
                #print("Warning: data less than cached data which doesn't make sense unless participating chiplets change in which case cached data needs to change")
                #print(f"\t{kernel_shard['kernel']} - {perf[1]['data']}")
                perf[1]["data"] = 0

        # always need to write back data if kernel isn't fused
        # use memory as "broadcast mechanism" 
        # can always save some data in SRAM for next kernel 
        self.cached_data = min(perf[1]["out"], self.get_on_chip_capacity())

        # If the wl kernels are small enough, they could be cached entirely and so need to consider steady state of model in some cases
        if self.warmup:
            self.max_cached_data = max(self.max_cached_data, self.cache_buf_scalar * (self.cached_data + perf[1]["weight"]))
        else:
            #if self.chiplet_id == 0:
            #    print("\tavailable cache", self.available_cache/(1024**2))
            sub_weight = min(self.available_cache, perf[1]["weight"])
            self.available_cache -= sub_weight
            perf[1]["weight"] -= sub_weight

        if fuse_next:
            perf[1]["out"] *= (self.cached_leak)

        # don't return the final performance here  - only the data accessed / flops calculated so the system analytical model can derive how much effective BW this chiplet gets
        # save intermediate paramters for final calculations
        if verbose > 0 and self.chiplet_id == 0:
            self.print_analytical_perf(perf)

        self.kernel_name        = kernel_shard["kernel"]
        self.total_kernel_data  = total_kernel_data
        self.analytical_perf    = perf
        self.dtype              = dtype
        return perf

    
    def get_perf(self, chiplet_data, mem_bw_g, verbose):
        # chiplets can override kernel to integrate level 3 or more accurate level 1 kernel
        # derive latency performance - use roofline (bounded by system bandwidth allocation) to calculate performance
        effective_bw = min(mem_bw_g, self.min_chiplet_bw_g)
        
        if verbose > 0 and self.chiplet_id == 0:
            print()
            print("\tMem BW: \t\t%0.2f GB/sec" % (mem_bw_g/(1024**3)))
            #print("\tChiplet BW: \t\t%0.2f GB/sec" % (self.chiplet_bw))
            print("\tBW Saturation: \t\t%0.2f GB/sec" % (self.bw_sat))
            print()

        flops = self.analytical_perf[0]
        mem_kernel_data = chiplet_data

        compute_exe = flops / self.get_flops(self.kernel_name)

        memory_exe = mem_kernel_data / effective_bw
                
        exe_time = compute_exe if compute_exe > memory_exe else memory_exe

        if verbose > 0: #and self.chiplet_id == 0
            print("\t%s\t%s\t(%0.2fus %s %0.2fus)" % (self.name.capitalize(), "Compute Bound" if (compute_exe > memory_exe) else "Memory Bound", compute_exe*(10**6), ">" if (compute_exe > memory_exe) else "<", memory_exe*(10**6)))
        
        def f_exp(x):
            return -1 * (x-1)**2 + 1

        energy = exe_time * (self.tdp_compute * f_exp(compute_exe/exe_time) + self.tdp_mem_controllers * f_exp(memory_exe/exe_time) + self.tdp_base)
        
        return {"exe_time": exe_time, "energy": energy}











    #################################
    #################################
    # kernel analysis below 
    #################################
    #################################

    def linear(self, kernel):
        batch_size      = kernel["input"]["batch-size"]
        hidden_dim      = kernel["input"]["hidden-dim"]
        out_dim         = kernel["output"]["output-dim"]

        def calculate_flops(kernel):   
            flops = (2 * batch_size * hidden_dim * out_dim)
            return flops

        def calculate_memory_access(kernel):       
            data_read = batch_size * (hidden_dim)
            weight_read = hidden_dim * (out_dim) + out_dim
            out_write = batch_size * out_dim 
            return {"data": data_read, "weight": weight_read, "out": out_write}
        
        return calculate_flops(kernel), calculate_memory_access(kernel)

    # https://stackoverflow.com/questions/27728531/convolutional-neural-networks-multiple-channels
    def conv_2d(self, kernel):
        # decode important data from kernel
        batch_size      = kernel["input"]["batch-size"]
        in_channels     = kernel["input"]["in-channels"]
        img_height      = kernel["input"]["img-height"]
        img_width       = kernel["input"]["img-width"]

        out_channels    = kernel["output"]["out-channels"]
        out_img_height  = kernel["output"]["img-height"]
        out_img_width   = kernel["output"]["img-width"]
        
        kernel_size     = kernel["layer"]["kernel"] if len(kernel["layer"]["kernel"]) == 2 else [kernel["layer"]["kernel"], kernel["layer"]["kernel"]]
        stride_size     = kernel["layer"]["stride"] if len(kernel["layer"]["stride"]) == 2 else [kernel["layer"]["stride"], kernel["layer"]["stride"]]
        padding_size    = kernel["layer"]["padding"] if len(kernel["layer"]["padding"]) == 2 else [kernel["layer"]["padding"], kernel["layer"]["padding"]]

        def calculate_flops(kernel):
            # computation
            flops_kernel = 2 * kernel_size[0] * kernel_size[1]
            kernel_places = int((img_height + padding_size[0]*2 - kernel_size[0])/stride_size[0] + 1) * int((img_width + padding_size[1]*2 - kernel_size[1])/stride_size[1] + 1)

            # edge zeros
            per_row_zero_flops = 0
            for i, overhang in enumerate(reversed(range(1, padding_size[0]+1))):
                if (i % stride_size[0]) == 0:
                    per_row_zero_flops += kernel_size[1] * overhang
            per_row_zero_flops *= (2 * (img_height-kernel_size[0]+1)/stride_size[1]) # assume same number of zeros on opposite side - might be incorrect

            # top zeros
            if padding_size[0] == padding_size[1] and kernel_size[0] == kernel_size[1]:
                per_col_zero_flops = per_row_zero_flops
            else:
                per_col_zero_flops = 0
                for i, overhang in enumerate(reversed(range(1, padding_size[1]+1))):
                    if (i % stride_size[1]) == 0:
                        per_col_zero_flops += kernel_size[0] * overhang
                per_col_zero_flops *= (2 * (img_width-kernel_size[1]+1)/stride_size[0]) # assume same number of zeros on opposite side

            # corner zeros
            per_cor_zero_flops = 0
            for i, overhang_e in enumerate(reversed(range(1, padding_size[0]+1))):
                if (i % stride_size[0]) == 0:
                    for j, overhang_t in enumerate(reversed(range(1, padding_size[1]+1))):
                        if (j % stride_size[1]) == 0:
                            per_cor_zero_flops += kernel_size[0] * kernel_size[1] - (kernel_size[0] - overhang_e) * (kernel_size[1] - overhang_t)
            per_cor_zero_flops *= 4
                        
            padding_zeros = 2 * (per_row_zero_flops + per_col_zero_flops + per_cor_zero_flops)
            flops = (flops_kernel * kernel_places - padding_zeros) * in_channels * out_channels
            return batch_size * flops

        def calculate_memory_access(kernel):
            data_read = batch_size * in_channels * img_height * img_width
            weight_read = (kernel_size[0] * kernel_size[1]) * out_channels * in_channels
            out_write = batch_size * out_channels * out_img_height * out_img_width 
            return {"data": data_read, "weight": weight_read, "out": out_write}
        
        comp = calculate_flops(kernel)
        mem = calculate_memory_access(kernel)

        '''
        print()
        FLOPS = 30000
        BW = 64
        print("flops:\t\t%0.3f GFLOP" % (comp/(10**9)))
        compute_latency = ((comp)/(10**9) / FLOPS * 1000)
        print("compute_latency: %0.4fms" % compute_latency)
        #print("data:\t\t", mem["data"])
        #print("weight_read:\t", mem["weight"])
        #print("out_write:\t", mem["out"])
        mem_size = (mem["data"] + mem["weight"] + mem["out"]) / (1024**3)
        mem_latency = mem_size / (BW)*1000
        print("mem_latency:\t %0.4fms" % mem_latency)
        print("mem_size:\t %0.4fMB" % (2*mem_size*1024))
        print("compute bound: ", compute_latency>mem_latency)
        '''

        return comp, mem
    
    def embedding_bag_collection(self, kernel):
        batch_size      = kernel["input"]["batch-size"]
        num_tables      = kernel["input"]["num-tables"]
        num_lookups     = kernel["input"]["num-lookups"]
        embedding_dim   = kernel["layer"]["embedding-dim"]
        
        def calculate_flops(kernel):
            flops = batch_size * num_tables * num_lookups * embedding_dim       # sum each embedding 
            return flops

        def calculate_memory_access(kernel):    
            data_read = (batch_size * num_tables * num_lookups) + (batch_size * num_tables * num_lookups * embedding_dim)
            weight_read = 0
            out_write = batch_size * num_tables * embedding_dim
            return {"data": data_read, "weight": weight_read, "out": out_write}
        
        return calculate_flops(kernel), calculate_memory_access(kernel)
    
    def interaction_arch(self, kernel):
        batch_size      = kernel["input"]["batch-size"]
        dense_features  = kernel["input"]["dense-features"]
        num_tables      = kernel["input"]["num-tables"]
        embedding_dim   = kernel["input"]["embedding-dim"]
        sharding_flops  = kernel["input"]["sharding_dim"]
        diag_lower      = kernel["output"]["flat-diag-lower-dense"]
        
        def calculate_flops(kernel):
            flops = (2 * (num_tables + 1) * embedding_dim * (num_tables + 1)) * sharding_flops
            return flops

        def calculate_memory_access(kernel):    
            data_read = batch_size * (num_tables * embedding_dim + dense_features)
            weight_read = 0
            out_write = batch_size * diag_lower # diag_lower has concatenated dense_features already
            return {"data": data_read, "weight": weight_read, "out": out_write}
        
        return calculate_flops(kernel), calculate_memory_access(kernel)
    
    def attention(self, kernel):
        # beam size is 0 for first token 
        # seq-len is 0 for second token 

        batch_size      = kernel["input"]["batch-size"]
        beam_size       = kernel["input"]["beam-size"]
        seq_len         = kernel["input"]["seq-len"]
        model_dim       = kernel["input"]["hidden-dim"]     # model dimmension (input dim)
        kv_cache        = kernel["input"]["kv_cache"] if "kv_cache" in kernel["input"].keys() else kernel["input"]["kv-cache"]       # data which was precacluated - indicates second token
        
        num_heads       = kernel["layer"]["num-heads"]      # number of attention heads
        output_dim      = kernel["layer"]["output-dim"]     # size of hidden dimension for QKV
        
        hidden_dim      = int(output_dim / num_heads) if num_heads > 0 else 0

        if kv_cache > 0:
            seq_len = beam_size
            seq_len_2 = kv_cache
        else:
            seq_len_2 = seq_len

        def calculate_flops(kernel):
            # Create Q, K, and V using sequence length and batch size
            flops_qkvo = batch_size * 4 * (2 * seq_len * model_dim * hidden_dim) * num_heads
            flops_sim = batch_size * (2 * seq_len * hidden_dim * seq_len_2) * num_heads
            flops_softmax = batch_size * (32 * seq_len * seq_len_2) * num_heads
            flops = flops_qkvo + flops_sim + flops_softmax
            return flops

        def calculate_memory_access(kernel):    
            data_input      = batch_size * seq_len * model_dim
            kv_cache_r      = 2 * batch_size * kv_cache * (hidden_dim * num_heads) 

            weights_qkvo    = 4 * model_dim * hidden_dim * num_heads
            kv_cache_w      = 2 * batch_size * seq_len * (hidden_dim * num_heads)
            data_output     = batch_size * seq_len * (hidden_dim * num_heads)

            data_read = data_input
            weight_read = weights_qkvo + kv_cache_r
            out_write = data_output + kv_cache_w
            return {"data": data_read, "weight": weight_read, "out": out_write}
        
        return calculate_flops(kernel), calculate_memory_access(kernel)


    def gcn_conv(self, kernel):
        debug = 0
        num_nodes = kernel["input"]["num-nodes"]
        num_edges = kernel["input"]["num-edges"]
        curve_fit = kernel["input"]["curve-fit-vals"]["1"][0]      # one cut meaning the full graph
        embedding_dim = kernel["output"]["embedding-dim"]
        sharding_dim = kernel["layer"]["sharding_dim"]

        cache_size = self.get_on_chip_capacity()
        max_cached_embeddings = int(cache_size / embedding_dim / self.dtype)
        previously_cached = int(self.cached_data / embedding_dim / self.dtype)
        num_cached_embeddings = int(min(max_cached_embeddings, num_nodes))

        # Define a long-tail distribution function for approximation
        def long_tail_func(x, a, b, c, d, e):
            return a * np.exp(-b * x ) + c*x + d*(x**2) + e

        # try to save graph to database if possible
        if "gcn" not in self.database:
            self.database["gcn"] = {}
        
        gcn_index = "-".join([str(curve_fit[0]), str(curve_fit[1]), str(curve_fit[2]), str(curve_fit[3]), str(curve_fit[4])])
        if gcn_index not in self.database["gcn"]:
            all_ids = np.linspace(1, int(num_nodes), int(num_nodes)-1)
            log_ids = np.log(all_ids)
            node_access_frequency = long_tail_func(log_ids, curve_fit[0], curve_fit[1], curve_fit[2], curve_fit[3], curve_fit[4]) 
            node_sum_total = np.sum(node_access_frequency)
            self.database["gcn"][gcn_index] = [node_access_frequency, node_sum_total]
        else:
            node_access_frequency, node_sum_total = self.database["gcn"][gcn_index]

        if debug:
            print("num cached embeddings", max_cached_embeddings)
            print("previous layer cached embeddings", previously_cached)

        def calculate_flops(kernel):   
            flops = 2 * num_edges * embedding_dim  # val * edge + node (MAC op)
            return flops

        def calculate_memory_access(kernel): 
            data_read = num_edges + num_edges * embedding_dim
            weight_read = 0
            out_write = num_nodes * embedding_dim
            
            # cached ids
            if debug:
                cached_ids = np.linspace(1, int(num_cached_embeddings), int(num_cached_embeddings))
                log_cached_ids = np.log(cached_ids)
                node_access_frequency_cached = long_tail_func(log_cached_ids, curve_fit[0], curve_fit[1], curve_fit[2], curve_fit[3], curve_fit[4])
                emb_memory_accesses_cached = np.sum(node_access_frequency_cached) * sharding_dim
            
            # speed this up
            # uncached ids
            if debug:
                uncached_ids = np.linspace(num_cached_embeddings, int(num_nodes), int(num_nodes-num_cached_embeddings))
                log_uncached_ids = np.log(uncached_ids)
                node_access_frequency_uncached = long_tail_func(log_uncached_ids, curve_fit[0], curve_fit[1], curve_fit[2], curve_fit[3], curve_fit[4])            
                emb_memory_accesses_uncached = np.sum(node_access_frequency_uncached) * sharding_dim

            emb_memory_accesses_uncached = (node_sum_total - np.sum(node_access_frequency[:num_cached_embeddings])) * sharding_dim

            # get first cache access for "frequently reused" embeddings if not cached from prior layer
            prev_cached_ids = np.linspace(1, num_nodes, previously_cached)
            cached_accesses_precached = len(np.where(prev_cached_ids < num_cached_embeddings)[0])
            first_access_then_cached = num_cached_embeddings - cached_accesses_precached
            
            if debug:
                print("previously cached: ", cached_accesses_precached)
                print("first access then cached: ", first_access_then_cached)

                print("cached ids: ", len(cached_ids), cached_ids)
                print("uncached ids: ", len(uncached_ids), uncached_ids)

                print("on-chip accesses", emb_memory_accesses_cached)
                print("off-chip embedding accesses", emb_memory_accesses_uncached)
                print("Total Accesses", (emb_memory_accesses_cached + emb_memory_accesses_uncached), " == ", num_edges)
            

            total_off_chip_embedding_accesses_reads = (first_access_then_cached + emb_memory_accesses_uncached) * embedding_dim # off chip accesses are sum of what wasn't cached from previous layer in "frequently reused" embeddings, and non-frequently reused embeddings
            csr_read = ((num_nodes * sharding_dim) + num_edges) * (4/self.dtype) + num_edges # (8/self.dtype) used to preadjust data type for fp32 (second num edges is the value pointer)
            csr_write = (num_nodes * sharding_dim) * embedding_dim

            # currently no good way to cache the CSR - so even if you had infinite memory, then you won't see the gains unless enough for N-layers * CSR size
            if debug: 
                print("Read CSR: %0.2f MB" % (csr_read*self.dtype/1024**2))

            data_read = total_off_chip_embedding_accesses_reads + csr_read
            weight_read = 0
            out_write = csr_write

            return {"data": data_read, "weight": weight_read, "out": out_write}
        
        return calculate_flops(kernel), calculate_memory_access(kernel)

    def add(self, kernel):
        elements = kernel["input"]["elements"]
        def calculate_flops(kernel):   
            flops = elements
            return flops

        def calculate_memory_access(kernel):       
            data_read = elements
            weight_read = 0
            out_write = elements
            return {"data": data_read, "weight": weight_read, "out": out_write}
        
        return calculate_flops(kernel), calculate_memory_access(kernel)

    def batch_norm(self, kernel):
        # decode important data from kernel
        batch_size      = kernel["input"]["batch-size"]
        in_channels     = kernel["input"]["in-channels"]
        img_height      = kernel["input"]["img-height"]
        img_width       = kernel["input"]["img-width"]

        out_channels    = kernel["output"]["out-channels"]
        out_img_height  = kernel["output"]["img-height"]
        out_img_width   = kernel["output"]["img-width"]

        num_features    = kernel["layer"]["features"]

        def calculate_flops(kernel):   
            flops = batch_size * 4 * num_features * (img_height * img_width)
            return flops

        def calculate_memory_access(kernel):       
            data_read = batch_size * in_channels * img_height * img_width
            weight_read = 2 * num_features
            out_write = batch_size * out_channels * out_img_height * out_img_width
            return {"data": data_read, "weight": weight_read, "out": out_write}
        
        return calculate_flops(kernel), calculate_memory_access(kernel)
    
    def group_norm(self, kernel):
        # decode important data from kernel
        batch_size      = kernel["input"]["batch-size"]
        in_channels     = kernel["input"]["in-channels"]
        img_height      = kernel["input"]["img-height"]
        img_width       = kernel["input"]["img-width"]

        out_channels    = kernel["output"]["out-channels"]
        out_img_height  = kernel["output"]["img-height"]
        out_img_width   = kernel["output"]["img-width"]

        num_features    = kernel["layer"]["num_channels"]
        num_groups      = kernel["layer"]["num_groups"]

        def calculate_flops(kernel):   
            flops = batch_size * 9 * num_features * (img_height * img_width)
            return flops

        def calculate_memory_access(kernel):       
            data_read = batch_size * in_channels * img_height * img_width
            weight_read = 2 * num_features
            out_write = batch_size * out_channels * out_img_height * out_img_width
            return {"data": data_read, "weight": weight_read, "out": out_write}
        
        return calculate_flops(kernel), calculate_memory_access(kernel)
    
    def layer_norm(self, kernel):
        # beam size is 0 for first token 
        # seq-len is 0 for second token 

        batch_size      = kernel["input"]["batch-size"]
        beam_size       = kernel["input"]["beam-size"]
        seq_len         = kernel["input"]["seq-len"]
        model_dim       = kernel["input"]["hidden-dim"]     # model dimmension (input dim)
        
        weights         = kernel["layer"]["weights"]        # number of normalization parameters in layer norm
        
        if beam_size > 0:
            seq_len = beam_size

        def calculate_flops(kernel):   
            flops = 11 * seq_len * model_dim
            return flops

        def calculate_memory_access(kernel):       
            data_io        = batch_size * seq_len * model_dim
            
            data_read = data_io
            weight_read = weights
            out_write = data_io
            return {"data": data_read, "weight": weight_read, "out": out_write}
        
        return calculate_flops(kernel), calculate_memory_access(kernel)

    def relu(self, kernel):
        elements = kernel["input"]["elements"]
        def calculate_flops(kernel):   
            flops = elements
            return flops

        def calculate_memory_access(kernel):       
            data_read = elements
            weight_read = 0
            out_write = elements
            return {"data": data_read, "weight": weight_read, "out": out_write}
        
        return calculate_flops(kernel), calculate_memory_access(kernel)
    
    def gelu(self, kernel):
        elements = kernel["input"]["elements"]
        def calculate_flops(kernel):   
            flops = 38 * elements
            return flops

        def calculate_memory_access(kernel):       
            data_read = elements
            weight_read = 0
            out_write = elements
            return {"data": data_read, "weight": weight_read, "out": out_write}
        
        return calculate_flops(kernel), calculate_memory_access(kernel)
    
    def silu(self, kernel):
        elements = kernel["input"]["elements"]
        def calculate_flops(kernel):   
            flops = 23 * elements
            return flops

        def calculate_memory_access(kernel):       
            data_read = elements
            weight_read = 0
            out_write = elements
            return {"data": data_read, "weight": weight_read, "out": out_write}
        
        return calculate_flops(kernel), calculate_memory_access(kernel)
    
    def softmax(self, kernel):
        elements = kernel["input"]["elements"]
        def calculate_flops(kernel):   
            flops = 25 * elements
            return flops

        def calculate_memory_access(kernel):       
            data_read = elements
            weight_read = 0
            out_write = elements
            return {"data": data_read, "weight": weight_read, "out": out_write}
        
        return calculate_flops(kernel), calculate_memory_access(kernel)

    def max_pool_2d(self, kernel):
        # decode important data from kernel
        batch_size      = kernel["input"]["batch-size"]
        in_channels     = kernel["input"]["in-channels"]
        img_height      = kernel["input"]["img-height"]
        img_width       = kernel["input"]["img-width"]

        out_channels    = kernel["output"]["out-channels"]
        out_img_height  = kernel["output"]["img-height"]
        out_img_width   = kernel["output"]["img-width"]
        
        def calculate_flops(kernel):   
            return 0

        def calculate_memory_access(kernel):       
            data_read = batch_size * (in_channels * img_height * img_width)
            weight_read = 0
            out_write = batch_size * (in_channels * out_img_width * out_img_height)
            return {"data": data_read, "weight": weight_read, "out": out_write}
        
        return calculate_flops(kernel), calculate_memory_access(kernel)

    def adaptive_avg_pool_2d(self, kernel):
        # decode important data from kernel
        batch_size      = kernel["input"]["batch-size"]
        in_channels     = kernel["input"]["in-channels"]
        img_height      = kernel["input"]["img-height"]
        img_width       = kernel["input"]["img-width"]

        out_channels    = kernel["output"]["out-channels"]
        out_img_height  = kernel["output"]["img-height"]
        out_img_width   = kernel["output"]["img-width"]

        def calculate_flops(kernel):   
            flops = batch_size * in_channels * img_height * img_width
            return flops

        def calculate_memory_access(kernel):       
            data_read = batch_size * in_channels * img_height * img_width
            weight_read = 0
            out_write = batch_size * (in_channels * out_img_width * out_img_height)
            return {"data": data_read, "weight": weight_read, "out": out_write}
        
        return calculate_flops(kernel), calculate_memory_access(kernel)


        