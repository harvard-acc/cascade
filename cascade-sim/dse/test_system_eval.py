import os
import time
import argparse
import matplotlib 
matplotlib.use('agg')

import statistics
import numpy as np
from scipy.optimize import minimize, LinearConstraint

from lib.plot_results import PlotResults
from lib.chiplet_system import ChipletSystem
from lib.trace_parser import TraceParser


###################################################
# hyperparameters for chiplet selection
GPU     = 12
ATTEN   = 0
SPARSE  = 0
CONV    = 0
NUM_CHANNELS = 16   # Max number of Memory Channels
###################################################

if __name__ == "__main__":  
    print("Starting")  
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--chiplet-library', type=str)
    parser.add_argument('--trace-dir', type=str)
    parser.add_argument('--experiment', type=str)
    args = parser.parse_args()

    tp = TraceParser(args.trace_dir, args.experiment)

    desired_chiplets = ["gpu"] * GPU + ["atten"] * ATTEN + ["sparse"] * SPARSE + ["conv"] * CONV

    # warmup sets the cache correctly so not-necessarily just streaming weights - cut_dim = "weights" or "batch"
    agg_kernel_results = []
    for TRACE_ID in range(len(tp.all_traces)):
        tp.all_traces[TRACE_ID].print_line()
        print("Working on Trace %i" % TRACE_ID)

        # create SoC and add chiplets to SoC
        cs = ChipletSystem(args.chiplet_library, verbose=1)
        cs.configure_system(desired_chiplets, tp.optimization_goal)
        cs.init_system_bandwidth(bw_per_channel=64, num_channels=NUM_CHANNELS)
        kernel_results = cs.characterize_workload(tp.get_trace(TRACE_ID), cut_dim="batch" if tp.all_traces[TRACE_ID].get_model() == "dnn" else "weights", dtype=2)
        agg_kernel_results += kernel_results * tp.all_traces[TRACE_ID].weighted_score # sudo run the workload "weighted_score" times
    

    pr = PlotResults("./results")
    pr.plot_kernel_roofline(tp.get_trace(TRACE_ID).get_model_instance(), tp.get_trace(TRACE_ID).get_batch_size(), agg_kernel_results, cs.compute_roofline(), cs.bandwidth_roofline())
    pr.plot_work_breakdown(tp.get_trace(TRACE_ID).get_model_instance(), tp.get_trace(TRACE_ID).get_batch_size(), agg_kernel_results, cs.get_num_chiplets())

