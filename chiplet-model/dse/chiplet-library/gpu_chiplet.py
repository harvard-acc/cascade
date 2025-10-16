import os

from base_chiplet import BaseChiplet

class GPUChiplet(BaseChiplet):
    def __init__(self, chiplet_id=None):
        super().__init__(chiplet_id=chiplet_id)
        self.name           = "gpu"
        self.bf16_flops     = 10000     # gflops    
        self.bw_sat         = 64        # gb/sec    
        self.tdp            = 15        # W         
        self.area           = 25        # mm^2      
        self.sram           = 4         # MB        
        self.kernels        = ["all"]

        self.finish_init()
        

        