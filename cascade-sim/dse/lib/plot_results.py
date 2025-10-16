import os
import json
import numpy as np
import pandas as pd
import matplotlib 
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

'''
import sys
sys.path.append("/nfs/home/matthewa/workspace/ml-characterization/level-3-kernel-hooks/chiplet-actuary")

from typing import Tuple
import math
from ChipletHL import module
from ChipletHL import chip
from ChipletHL import package
from ChipletHL import utils
from ChipletHL import spec
'''

GB_PER_DIMM = 8
DRAM_COST_PER_GB = 1.5 
DRAM_POWER = 3  # Watts / 8GB DIMM

class PlotResults:
    def __init__(self, results_dir):
        self.title_size = 12
        self.axtitle_size = 10
        self.xlabel_size = 10
        self.ylabel_size = 10
        self.xtick_size = 8
        self.ytick_size = 8
        self.legend_size = 10
        
        self.marker_size = 6
        self.marker_edge_width = .8

        self.large_marker_size = 8
        self.small_marker_size = 6

        self.dpi = 100
        
        self.results_dir = results_dir
        
        #self.my_color_map = ["#A51C30", "#3872B2", "#EC8F9C", "#FF6600", "#ECDD7B", "#808080"]
        self.my_color_map = ["#ACC3B1", "#B7D1E2", "#F1A151", "#F1D365", "#DAD7D0", "#668C87"]

    def plot_kernel_roofline(self, model_instance, batch_size, kernel_results, compute_roofline, bw_roofline):
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))

        #####################################################################################
        compute_roofline = compute_roofline
        bw_roofline = bw_roofline
        overlap_index = compute_roofline / bw_roofline

        all_names = []
        all_flop = []
        all_mem = []
        all_exe = []
        for i, kernel in enumerate(kernel_results):
            all_names.append(kernel["name"])
            all_flop.append(kernel["total"]["flops"])
            all_mem.append(kernel["total"]["mem_accessed"])
            all_exe.append(kernel["total"]["exe_time"])            

        for i, kernel in enumerate(kernel_results):
            flop = all_flop[i]
            mem_accessed = all_mem[i]
            exe_time = all_exe[i]
            
            ai = flop / mem_accessed
            throughput = flop / exe_time / (10**9) # GFLOPs
            
            fraction_lagest_exe = (exe_time - min(all_exe))/(max(all_exe)-min(all_exe)) if len(all_exe) > 1 else 1
            
            # annotate kernels which are more than x% of largest single-kernel execution time
            if fraction_lagest_exe > .01:
                ax.text(ai, throughput, all_names[i], va="bottom", ha="left", rotation=20, zorder=1)

            ax.plot(ai, throughput, "X", markersize=4+12*fraction_lagest_exe, markerfacecolor=self.my_color_map[0], markeredgecolor="black", zorder=0)

        agg_ai = sum(all_flop) / sum(all_mem)
        agg_throughput = sum(all_flop) / sum(all_exe) / (10**9)
        ax.plot(agg_ai, agg_throughput, "o", markersize=12, label="Full System", markerfacecolor=self.my_color_map[0], markeredgecolor="black")

        # plot roofline
        ai = np.arange(0, max(max(np.array(all_flop) / (np.array(all_mem))), overlap_index*2)*1.1, 1)
        comp_throughput = compute_roofline * np.ones(len(ai))
        bw_throughput = bw_roofline * (1024**3) / (10**9) * np.arange(len(ai))
        throughput = np.minimum(comp_throughput, bw_throughput)
        ax.plot(ai, throughput, color="black")

        # Labels and legend
        ax.set_xscale("log", base=10)
        ax.set_yscale("log", base=10)

        ax.set_xlabel("Arithmetic Intensity (FLOPs/Byte)", fontsize=self.xlabel_size)
        ax.set_ylabel("Throughput (FLOPS)", fontsize=self.ylabel_size)
        ax.set_title(f"Roofline Analysis Chiplet System", fontsize=self.title_size)
        ax.legend(title="", loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=self.legend_size)
        #ax.rcParams['legend.title_fontsize'] = self.legend_size

        print(f"Run Time:\t{sum(all_exe)*1000} ms")
    
        plt.tight_layout()
        plt.savefig(self.results_dir + f"/roofline-{model_instance}-{batch_size}-bs-kernels.pdf", dpi=self.dpi)
        plt.clf()
        plt.close()
    

    def plot_work_breakdown(self, model_instance, batch_size, kernel_results, num_chiplets):
        kernel_names = [] 
        kernel_exe = []
        kernel_energy = []
        kernel_work = {}
        kernel_breakdown = {}
        chiplet_names = []
        for chiplet_id in range(num_chiplets):
            kernel_work[chiplet_id] = []
            chiplet_names.append("")

        for kernel in kernel_results:
            for chiplet_id in range(num_chiplets):
                if chiplet_id in kernel["chiplets"].keys(): # chiplet participated 
                    kernel_work[chiplet_id].append(kernel["chiplets"][chiplet_id]["work"])
                    chiplet_names[chiplet_id] = kernel["chiplets"][chiplet_id]["name"]
                else: # chiplet did not participate
                    kernel_work[chiplet_id].append(0)
            kernel_exe.append(kernel["total"]["exe_time"])
            kernel_energy.append(kernel["total"]["energy"] )

            if kernel["name"] not in kernel_breakdown:
                kernel_breakdown[kernel["name"]] = 0
            kernel_breakdown[kernel["name"]] += kernel["total"]["exe_time"]
            kernel_names.append(kernel["name"])
        
        # normalize kernel_work
        total_exe = sum(kernel_exe)
        total_energy = sum(kernel_energy)

        kernel_work_agg = []
        for chiplet_id in range(num_chiplets):
            kernel_work_agg.append(np.dot(np.array(kernel_work[chiplet_id]), np.array(kernel_exe)/total_exe))
        
        print(f"{len(kernel_work_agg)} frac work per chiplet: [", end=" ")
        for frac_work in kernel_work_agg:
            print("%0.2f%%" % (frac_work*100), end=" ")
        print("]")
        print("Total Time: %0.5fms" % (total_exe*1000))
        print("Total Energy: %0.5fmJ" % (total_energy*10**3))

        #####################################################################################
        #####################################################################################
        # kernel-exe breakdown
        labels = kernel_breakdown.keys()
        sizes = kernel_breakdown.values()


        fig, ax = plt.subplots(1, 1, figsize=(7, 4))

        # Create a pie chart
        wedges, texts, autotexts = ax.pie(sizes, labels=None, colors=self.my_color_map, autopct='%1.1f%%', startangle=0, textprops=dict(color="w"))
        ax.legend(wedges, labels, title='Kernels', loc='center left', bbox_to_anchor=(1, 0, 0.5, 1))
        plt.setp(autotexts, size=self.title_size, weight="bold")

        # Equal aspect ratio ensures that pie is drawn as a circle
        plt.axis("equal")  
        plt.title(f"Kernel Breakdown - {model_instance} - BS={batch_size}", fontsize=self.title_size)

        plt.tight_layout()
        plt.savefig(self.results_dir + f"/kernel-breakdown-workload-{model_instance}-{batch_size}-bs.pdf", dpi=self.dpi)
        plt.clf()
        plt.close()
        #####################################################################################
        #####################################################################################



        #####################################################################################
        #####################################################################################
        # chiplet work plot
        labels = chiplet_names
        sizes = kernel_work_agg


        fig, ax = plt.subplots(1, 1, figsize=(7, 4))

        # Create a pie chart
        wedges, texts, autotexts = ax.pie(sizes, labels=None, colors=self.my_color_map, autopct='%1.1f%%', startangle=0, textprops=dict(color="w"))
        ax.legend(wedges, labels, title='Chiplets', loc='center left', bbox_to_anchor=(1, 0, 0.5, 1))
        plt.setp(autotexts, size=self.title_size, weight="bold")

        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.set_ylabel("Fraction Work", fontsize=self.ylabel_size)
        plt.axis("equal")  
        plt.title(f"Chiplet Work Breakdown - {model_instance} - BS={batch_size}", fontsize=self.title_size)

        plt.tight_layout()
        plt.savefig(self.results_dir + f"/kernel-breakdown-chiplets-{model_instance}-{batch_size}-bs.pdf", dpi=self.dpi)
        plt.clf()
        plt.close()
        #####################################################################################
        #####################################################################################


        #####################################################################################
        #####################################################################################
        # per-kernel-breakdown         
        labels = chiplet_names
        sizes = kernel_work_agg

        fig, ax = plt.subplots(1, 1, figsize=(7, 4))

        if model_instance == "stable-diffusion-v1-4":
            START_BAR = 11
            NUM_KERNELS = 20
            self.my_color_map[1] = self.my_color_map[3]
        else:
            START_BAR = 0
            NUM_KERNELS = 20

        # Create a pie chart
        width = 1/(len(chiplet_names)+1)
        x_pos = np.arange(len(kernel_work[0][START_BAR:NUM_KERNELS]))
        for chiplet_id, chiplet_name in enumerate(chiplet_names):
            ax.bar(x_pos+chiplet_id*width, kernel_work[chiplet_id][START_BAR:NUM_KERNELS], width, color=self.my_color_map[chiplet_id % len(self.my_color_map)], label=chiplet_name)

        ax.set_ylim(0, 1)
        ax.set_ylabel("Fraction of Compute per Chiplet")
        ax.set_xticks(x_pos+(width*len(chiplet_names))/2 - width/2)
        ax.set_xticklabels(kernel_names[START_BAR:NUM_KERNELS], rotation=45, ha="right")
        ax.legend()
        # Equal aspect ratio ensures that pie is drawn as a circle
        plt.title(f"Chiplet Kernel Breakdown - {model_instance} - BS={batch_size}", fontsize=self.title_size)

        plt.tight_layout()
        plt.savefig(self.results_dir + f"/kernel-breakdown-chiplets-kernels-{model_instance}-{batch_size}-bs.pdf", dpi=self.dpi)
        plt.clf()
        plt.close()
        #####################################################################################
        #####################################################################################
