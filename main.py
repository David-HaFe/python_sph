# main file for my python code

import argparse
import sys

import heat_equation
import heat_equation_analytical
import navier_stokes_incompressible
import navier_stokes_compressible

from utils.diagnostics import diagnostics

parser = argparse.ArgumentParser()
parser.add_argument("--run", help="which example to run")

parser.add_argument(
    "--no_plot",
    action="store_true",
    help="if plot/animation should be drawn",
)
parser.add_argument(
    "--no_csv",
    action="store_true",
    help="if csv should be generated",
)

args = parser.pars_args()

for run in args.run:
    if mode == "heat_eqaution":

    elif mode == "heat_equation_analytical":

    elif mode == "navier_stokes_incompressible":

    elif mode == "navier_stokes_compressible":

    else:
        print("This is not a valid program, now you have to implement it"
        sys.exit()

if not args.no_plot:

if not args.no_csv:

