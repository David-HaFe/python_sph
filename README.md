# Python fluid sim
Python implementation of the fluid simulation method for my master's thesis.

## Usage
If you don't know what you are doing, you should probably only interact
with the `config.py` file. Here you can define everything from model parameters
to solver and simulation grid setup.
After you are happy with the settings, you can run a simulation via the command
```
python3 main.py
```
and one of the following **mandatory** flags:
```
:construction: -- compare -> calculates a table of the MSE between all runs specified in config.py
```
```
:construction: -- visualize kernel -> plots a kernel function that was specified in config.py
```
```
-- run -> runs a simulation of the specified example
```
valid options for `--run` are

|  | name | description |
| :white_check_mark: | heat_equation | simulating heat transfer using a PDE |
| :white_check_mark: | heat_equation_analytical | evaluating a known analytical solution of the heat equation at all simulation time steps |
| :warning: | navier_stokes_compressible | simulating compressible fluid flow (does weird things and is also deprecated) |
| :construction: | navier_stokes_incompressible | simulating incompressible fluid flow (not working) |



## Dependencies
In order to run this code, you will need python3, and a virtual environment,
which is created with
```
python -m venv .venv
```
and activated with
```
source .venv/bin/activate
```
Once this is done, you will need to install the packages

* numpy
* matplotlib
* scipy
* pandas
* playsound3 (if you want the sound effect when the simulation is done)

which can be done via
```
pip install numpy
pip install matplotlib
pip install scipy
pip install pandas
pip install playsound3
```


