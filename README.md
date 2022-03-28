# FNO4CO2

This repository contains some re-implementation of Fourier Neural Operator from [Fourier Neural Operator for Parameter Partial Differential Equations](https://arxiv.org/abs/2010.08895) authored by Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, et al. The [original repository](https://github.com/zongyi-li/fourier_neural_operator) is in python.

This code base is using the Julia Language and [DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named
> FNO

To (locally) reproduce this project, do the following:

0. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently.
1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box.

## Examples

The repository currently includes 2 examples. `fourier_2d.jl` trains a 2D FNO which maps the coefficients of Poisson equation to its solution. `fourier_3d.jl` trains a 2.5D FNO which maps the permeability to time-varying CO2 concentration in 2-phase flow.

## Citation

If you find this software useful in your research, we would appreciate it if you cite:

```bibtex
@unpublished {yin2022SEGlci,
	title = {Learned coupled inversion for carbon sequestration monitoring and forecasting with Fourier neural operators},
	year = {2022},
	note = {Submitted},
	month = {03},
	abstract = {Seismic monitoring of carbon storage sequestration is a challenging problem involving both fluid-flow physics and wave physics. Additionally, monitoring usually requires the solvers for these physics to be coupled and differentiable to effectively invert for the subsurface properties of interest. To drastically reduce the computational cost, we introduce a learned coupled inversion framework based on the wave modeling operator, rock property conversion and a proxy fluid-flow simulator. We show that we can accurately use a Fourier neural operator as a proxy for the fluid-flow simulator for a fraction of the computational cost. We demonstrate the efficacy of our proposed method by means of a synthetic experiment. Finally, our framework is extended to carbon sequestration forecasting, where we effectively use the surrogate Fourier neural operator to forecast the CO2 plume in the future at near-zero additional cost.},
	keywords = {CCS, deep learning, Fourier neural operators, inversion, machine learning, multiphysics, time-lapse},
	url = {https://slim.gatech.edu/Publications/Public/Submitted/2022/yin2022SEGlci/paper.html},
	software = {https://github.com/slimgroup/FNO4CO2},
	author = {Ziyi Yin and Ali Siahkoohi and Mathias Louboutin and Felix J. Herrmann}
}
```