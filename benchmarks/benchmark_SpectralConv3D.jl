# author: Ziyi (Francis) Yin
# This script benchmarks the time and memory usage for spectral conv layer in FNO
# The PDE is in 2D while FNO requires 3D FFT
# Date: Aug 2021

using DrWatson
@quickactivate "FNO"
import Pkg; Pkg.instantiate()

using PyPlot
using BSON
using Flux, Random, FFTW, Zygote, NNlib
using MAT, Statistics, LinearAlgebra
using CUDA
using ProgressMeter
using BenchmarkTools

try
    CUDA.device()
    global gpu_flag=true
catch e
    println("CUDA.device() found no GPU device on this machine.")
    global gpu_flag=false
end

CUDA.culiteral_pow(::typeof(^), a::Complex{Float32}, b::Val{2}) = real(conj(a)*a)
CUDA.sqrt(a::Complex) = cu(sqrt(a))
Base.broadcasted(::typeof(sqrt), a::Base.Broadcast.Broadcasted) = Base.broadcast(sqrt, Base.materialize(a))

include("../utils.jl")

Random.seed!(3)

mutable struct SpectralConv3d_fast{T,N}
    weights1::AbstractArray{T,N}
    weights2::AbstractArray{T,N}
    weights3::AbstractArray{T,N}
    weights4::AbstractArray{T,N}
end

@Flux.functor SpectralConv3d_fast

# Constructor
function SpectralConv3d_fast(in_channels::Integer, out_channels::Integer, modes1::Integer, modes2::Integer, modes3::Integer)
    scale = (1f0 / (in_channels * out_channels))
    if gpu_flag
        weights1 = scale*randn(Complex{Float32}, modes1, modes2, modes3, in_channels, out_channels) |> gpu
        weights2 = scale*randn(Complex{Float32}, modes1, modes2, modes3, in_channels, out_channels) |> gpu
        weights3 = scale*randn(Complex{Float32}, modes1, modes2, modes3, in_channels, out_channels) |> gpu
        weights4 = scale*randn(Complex{Float32}, modes1, modes2, modes3, in_channels, out_channels) |> gpu
    else
        weights1 = scale*rand(Complex{Float32}, modes1, modes2, modes3, in_channels, out_channels)
        weights2 = scale*rand(Complex{Float32}, modes1, modes2, modes3, in_channels, out_channels)
        weights3 = scale*rand(Complex{Float32}, modes1, modes2, modes3, in_channels, out_channels)
        weights4 = scale*rand(Complex{Float32}, modes1, modes2, modes3, in_channels, out_channels)
    end
    return SpectralConv3d_fast{Complex{Float32}, 5}(weights1, weights2, weights3, weights4)
end

function compl_mul3d(x::AbstractArray{Complex{Float32}}, y::AbstractArray{Complex{Float32}})
    # complex multiplication
    # x in (modes1, modes2, modes3, input channels, batchsize)
    # y in (modes1, modes2, modes3, input channels, output channels)
    # output in (modes1,modes2,modes3,output channels,batchsize)
    x_per = permutedims(x,[5,4,1,2,3]) # batchsize*in_channels*modes1*modes2*modes3
    y_per = permutedims(y,[4,5,1,2,3]) # in_channels*out_channels*modes1*modes2*modes3
    x_resh = reshape(x_per,size(x_per,1),size(x_per,2),:) # batchsize*in_channels*(modes1*modes2*modes3)
    y_resh = reshape(y_per,size(y_per,1),size(y_per,2),:) # in_channels*out_channels*(modes1*modes2*modes3)
    out_resh = batched_mul(x_resh,y_resh) # batchsize*out_channels*(modes1*modes2*modes3)
    out_per = reshape(out_resh,size(out_resh,1),size(out_resh,2),size(x,1),size(x,2),size(x,3)) # batchsize*out_channels*modes1*modes2*modes3
    out = permutedims(out_per,[3,4,5,2,1])
    return out
end

function (L::SpectralConv3d_fast)(x::AbstractArray{Float32})
    # x in (size_x, size_y, time, channels, batchsize)
    x_ft = rfft(x,[1,2,3])      ## full size FFT
    modes1 = size(L.weights1,1)
    modes2 = size(L.weights1,2)
    modes3 = size(L.weights1,3)
    ### only keep low frequency coefficients
    out_ft = cat(cat(cat(compl_mul3d(x_ft[1:modes1, 1:modes2, 1:modes3, :,:], L.weights1), 
                0f0im .* view(x_ft, 1:modes1, 1:modes2, 1:size(x_ft,3)-2*modes3, :, :),
                compl_mul3d(x_ft[1:modes1, 1:modes2, end-modes3+1:end,:,:], L.weights2),dims=3),
                0f0im .* view(x_ft, 1:modes1, 1:size(x_ft, 2)-2*modes2, :, :, :),
                cat(compl_mul3d(x_ft[1:modes1, end-modes2+1:end, 1:modes3,:,:], L.weights3),
                0f0im .* view(x_ft, 1:modes1, 1:modes2, 1:size(x_ft,3)-2*modes3, :, :),
                compl_mul3d(x_ft[1:modes1, end-modes2+1:end, end-modes3+1:end,:,:], L.weights4),dims=3)
                ,dims=2),
                0f0im .* view(x_ft, 1:size(x_ft,1)-modes1, :, :, :, :),dims=1)
    out_ft = irfft(out_ft, size(x,1),[1,2,3])
end

sizes = [2^i for i=5:7]
modes = [2^i for i=1:5]
width = [2^i for i=2:6]

batchsize = 16

time_ = zeros(Float32, length(sizes), length(modes), length(width))
memory_ = zeros(Float32, length(sizes), length(modes), length(width))

for (i, m)=enumerate(modes)
    for (j, w)=enumerate(width)
        global SC = SpectralConv3d_fast(w, w, m, m, m)
        Flux.trainmode!(SC, true)
        global weight_ = Flux.params(SC)
        for (k, s)=enumerate(sizes)
            global x = randn(Float32, s, s, s, w, batchsize)
            global y = randn(Float32, s, s, s, w, batchsize)
            if gpu_flag
                x = x |> gpu
                y = y |> gpu
            end
            B = (@benchmark grads = gradient(() -> Flux.mse(SC(x),y;agg=sum), weight_))
            time_[k, i, j] = mean(B).time/1f9
            memory_[k, i, j] = mean(B).memory
            println("SpectralConv3D test: $(m) modes, $(w) width, ($(s),$(s),$(s)) problem size in (x,y,t), $(batchsize) batchsize")
            println("Time = $(time_[k, i, j]) seconds")
            println("Memory = $(memory_[k, i, j]) Bytes")
            Base.flush(Base.stdout)
        end
    end
end

device_ = (gpu_flag ? "GPU" : "CPU")
JLD2.@save "BenchmarkSpectralConv3D_$(device_).jld2" time_ memory_ gpu_flag sizes modes width