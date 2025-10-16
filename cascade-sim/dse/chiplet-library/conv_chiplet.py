import os

from base_chiplet import BaseChiplet

class ConvChiplet(BaseChiplet):
    def __init__(self, chiplet_id=None):
        super().__init__(chiplet_id=chiplet_id)
        self.name           = "conv"    
        self.bf16_flops     = 30000     # gflops 
        self.bw_sat         = 64        # gb/sec 
        self.tdp            = 10        # W      
        self.area           = 25        # mm^2   
        self.sram           = 8         # MB     
        self.kernels        = ["all"] #["Conv2d", "BatchNorm", "ReLU", "MaxPool2d", "AdaptiveAvgPool2d", "Add", "Linear"]
        
        self.finish_init()
    
    def get_flops(self, kernel_name):
        if kernel_name in ["Conv2d"]:
            return self.bf16_flops_g
        else:
            return 7500 * (10**9)