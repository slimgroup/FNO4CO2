# Author: Ziyi Yin, ziyi.yin@gatech.edu

module FNO4CO2

export gpu_flag

using DrWatson
using Flux
using FFTW
using InvertibleNetworks:ActNorm
using CUDA

function __init__()
	global gpu_flag = parse(Bool, get(ENV, "FNO4CO2GPU", "0"))
	@info "FNO4CO2 is using " * (gpu_flag ? "GPU" : "CPU")
end

# Utilities.
include("./utils.jl")

# FNO model.
include("./FNOstruct.jl")

end
