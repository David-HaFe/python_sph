# main file for my python code

import argparse
import sys
from playsound3 import playsound

import heat_equation.main as heat_equation
import heat_equation_analytical.main as heat_equation_analytical
import navier_stokes_incompressible.main as navier_stokes_incompressible
import navier_stokes_compressible.main as navier_stokes_compressible

from utils.diagnostics import diagnostics
from utils.plot_particles import plot_particles
from utils.plot_temperature import (
    plot_temperature_map,
    plot_temperature_surface,
)
from utils.export_to_csv import export_to_csv

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

args = parser.parse_args()

if args.run == "heat_equation":
    t, x, y, T = heat_equation.main()
    if not args.no_plot:
        plot_temperature_map(t, x, y, T, "heat_equation")

elif args.run == "heat_equation_analytical":
    t, x, y, T = heat_equation_analytical.main()
    if not args.no_plot:
        plot_temperature_map(t, x, y, T, "heat_equation_analytical")

elif args.run == "navier_stokes_incompressible":
    navier_stokes_incompressible.main()

elif args.run == "navier_stokes_compressible":
    navier_stokes_compressible.main()

else:
    print("This is not a valid program, now you have to implement it")
    sys.exit()

# if not args.no_plot:
# plot_particles(t, x, y, is_border_particle)

if not args.no_csv:
    export_to_csv(t, T, "heat_eq")

print("")
diagnostics.print_diagnostics()
playsound("media/ding.wav")
