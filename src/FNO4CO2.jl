# Author: Ziyi Yin, ziyi.yin@gatech.edu

module FNO4CO2

export gpu_flag

using DrWatson
using Flux
using FFTW
using InvertibleNetworks:ActNorm
using CUDA

try
    @assert ENV["FNO4CO2GPU"] == "1"
    CUDA.device()
    global gpu_flag=true
    @info "using GPU for FNO4CO2"
catch e
    global gpu_flag=false
    @info "using CPU for FNO4CO2"
end

# Utilities.
include("./utils.jl")

# 3D FNO model.
include("./fno3dstruct.jl")

# sorry but there are currently some hacks
include("./hack.jl")

end