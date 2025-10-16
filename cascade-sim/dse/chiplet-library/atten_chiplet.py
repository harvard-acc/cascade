import os

from base_chiplet import BaseChiplet

class AttenChiplet(BaseChiplet):
    def __init__(self, chiplet_id=None):
        super().__init__(chiplet_id=chiplet_id)
        self.name           = "atten"
        self.bf16_flops     = 30000
        self.bw_sat         = 64
        self.tdp            = 15
        self.area           = 25
        self.sram           = 8 
        self.kernels        = ["all"] # ["Attention", "Linear", "LayerNorm", "GeLU"]

        self.finish_init()

    def get_flops(self, kernel_name):
        if kernel_name == "Attention":
            return self.bf16_flops_g
        else:
            return 7500 * (10**9)

        