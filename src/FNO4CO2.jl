# Author: Ziyi Yin, ziyi.yin@gatech.edu

module FNO4CO2

export gpu_flag

using DrWatson
using Flux
using FFTW
using InvertibleNetworks:ActNorm

try
    CUDA.device()
    global gpu_flag=true
catch e
    println("CUDA.device() found no GPU device on this machine.")
    global gpu_flag=false
end

# Utilities.
include("./utils.jl")

# 3D FNO model.
include("./fno3dstruct.jl")

end