import os

from base_chiplet import BaseChiplet

class SparseChiplet(BaseChiplet):
    def __init__(self, chiplet_id=None):
        super().__init__(chiplet_id=chiplet_id)
        self.name           = "sparse"
        self.bf16_flops     = 1000
        self.bw_sat         = 64*1.5
        self.tdp            = 4
        self.area           = 25
        self.sram           = 8
        #self.kernels        = ["Embedding", "EmbeddingTable", "GCNConv", "Linear", "ReLU", "Softmax", "Attention", "LayerNorm", "GeLU"]
        self.kernels        = ["all"]

        self.finish_init()

        