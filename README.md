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
