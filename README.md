# FNO4CO2

This repository contains the implementation of learned coupled inversion framework and the numerical experiments in [Learned coupled inversion for carbon sequestration monitoring and forecasting with Fourier neural operators](https://arxiv.org/abs/2203.14396).

This repository contains re-implementation of Fourier neural operators from [Fourier Neural Operator for Parameter Partial Differential Equations](https://arxiv.org/abs/2010.08895) authored by Zongyi Li et al. The [original repository](https://github.com/zongyi-li/fourier_neural_operator) is in python.

This code base is using the Julia Language and [DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named
> FNO

To (locally) reproduce this project, do the following:

1. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently.
2. Download [python](https://www.python.org/) and [Julia](https://julialang.org/). The numerical experiments are reproducible by python 3.7 and Julia 1.5.
3. Install [Devito](https://www.devitoproject.org/), a python package used for wave simulation.
4. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary Julia packages for you to be able to run the scripts and
everything should work out of the box.

## Examples

The repository currently includes 2 examples. `fourier_2d.jl` trains a 2D FNO which maps the coefficients of Poisson equation to its solution. `fourier_3d.jl` trains a 2.5D FNO which maps the permeability to time-varying CO2 concentration in 2-phase flow.

## Citation

If you find this software useful in your research, we would appreciate it if you cite:

```bibtex
@article{yin2022learned,
  title={Learned coupled inversion for carbon sequestration monitoring and forecasting with Fourier neural operators},
  author={Yin, Ziyi and Siahkoohi, Ali and Louboutin, Mathias and Herrmann, Felix J},
  journal={arXiv preprint arXiv:2203.14396},
  year={2022}
}
```

## Author

Ziyi (Francis) Yin, [ziyi.yin@gatech.edu](mailto:ziyi.yin@gatech.edu)