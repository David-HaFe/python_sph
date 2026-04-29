# main file for my python code

import argparse
import sys
from playsound3 import playsound
import numpy as np
from concurrent.futures import ProcessPoolExecutor

import heat_equation.main as heat_equation
import heat_equation_analytical.main as heat_equation_analytical
import navier_stokes_incompressible.main as navier_stokes_incompressible
import navier_stokes_compressible.main as navier_stokes_compressible

from utils.visualize_kernel import visualize_kernel
from utils.compare import compare_MSE, compare_scatter

from utils.diagnostics import diagnostics
from utils.plot_particles import plot_particles
from utils.plot_temperature import (
    plot_temperature_map,
    plot_temperature_surface,
)
from utils.export_to_npz import export_to_npz
from config import (
    sim_result,
    no_particles,
    no_steps,
    recompute,
)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--heat_equation",
    action="store_true",
)

parser.add_argument(
    "--heat_equation_analytical",
    action="store_true",
)

parser.add_argument(
    "--navier_stokes_compressible",
    action="store_true",
)

parser.add_argument(
    "--navier_stokes_incompressible",
    action="store_true",
)

parser.add_argument(
    "--no_plot",
    action="store_true",
    help="if plot/animation should be drawn",
)
parser.add_argument(
    "--no_npz",
    action="store_true",
    help="if npz should be generated",
)
parser.add_argument(
    "--compare_scatter",
    action="store_true",
    help="if true, display scatter plot of different files",
)
parser.add_argument(
    "--compare_mse",
    action="store_true",
    help="if true, print MSE table to compare npzs specified in config file",
)
parser.add_argument(
    "--visualize_kernel",
    action="store_true",
    help="if true, visualize the kernel specified in the config file",
)

# parser.add_argument(
#     "--particles",
#     action

args = parser.parse_args()

if args.heat_equation:
    sim_result = heat_equation.main()

    file_prefix = "heat_equation"
    if not args.no_plot:
        plot_temperature_map(sim_result, file_prefix)
        plot_temperature_surface(sim_result, file_prefix)

    if not args.no_npz:
        export_to_npz(sim_result, file_prefix)

if args.heat_equation_analytical:
    sim_result = heat_equation_analytical.main()

    file_prefix = "heat_equation_analytical"
    if not args.no_plot:
        plot_temperature_map(sim_result, file_prefix)
        plot_temperature_surface(sim_result, file_prefix)

    if not args.no_npz:
        export_to_npz(sim_result, file_prefix)

if args.navier_stokes_incompressible:
    t, x, y, is_border_particle = navier_stokes_incompressible.main()

    file_prefix = "navier_stokes_incompressible"
    if not args.no_plot:
        plot_particles(t, x, y, is_border_particle, file_prefix)

if args.navier_stokes_compressible:
    t, x, y, is_border_particle = navier_stokes_compressible.main()

    file_prefix = "navier_stokes_compressible"
    if not args.no_plot:
        plot_particles(t, x, y, is_border_particle, file_prefix)

# show kernel function, or whatever is thrown in there
if args.visualize_kernel:
    visualize_kernel()

if args.compare_scatter:
    compare_scatter()

if args.compare_mse:
    compare_MSE()


diagnostics.print_diagnostics()
playsound("misc/ding.wav")
