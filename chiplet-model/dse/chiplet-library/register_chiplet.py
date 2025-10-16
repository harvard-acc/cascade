import os

from base_chiplet import BaseChiplet
from gpu_chiplet import GPUChiplet
from sparse_chiplet import SparseChiplet
from conv_chiplet import ConvChiplet
from atten_chiplet import AttenChiplet


class RegisterChiplets:
    def __init__(self):
        self.chiplet_library = {}
    
    def register_chiplets(self):
        self.chiplet_library["gpu"]     = GPUChiplet
        self.chiplet_library["sparse"]  = SparseChiplet
        self.chiplet_library["conv"]    = ConvChiplet
        self.chiplet_library["atten"]   = AttenChiplet

        return self.chiplet_library
