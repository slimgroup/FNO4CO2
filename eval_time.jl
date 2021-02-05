# author: Ziyi Yin
# This code is an implementation of fourier neural operators from Zongyi Li's repository

using PyPlot
using BSON
using Flux, Random, FFTW, Zygote, NNlib
using MAT, Statistics, LinearAlgebra
using CUDA
using ProgressMeter, JLD2
using Images

CUDA.culiteral_pow(::typeof(^), a::Complex{Float32}, b::Val{2}) = real(conj(a)*a)
CUDA.sqrt(a::Complex) = cu(sqrt(a))
Base.broadcasted(::typeof(sqrt), a::Base.Broadcast.Broadcasted) = Base.broadcast(sqrt, Base.materialize(a))

include("utils.jl")

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
    #weights1 = scale*randn(Complex{Float32}, modes1, modes2, modes3, in_channels, out_channels) |> gpu
    #weights2 = scale*randn(Complex{Float32}, modes1, modes2, modes3, in_channels, out_channels) |> gpu
    #weights3 = scale*randn(Complex{Float32}, modes1, modes2, modes3, in_channels, out_channels) |> gpu
    #weights4 = scale*randn(Complex{Float32}, modes1, modes2, modes3, in_channels, out_channels) |> gpu
    weights1 = scale*rand(Complex{Float32}, modes1, modes2, modes3, in_channels, out_channels)
    weights2 = scale*rand(Complex{Float32}, modes1, modes2, modes3, in_channels, out_channels)
    weights3 = scale*rand(Complex{Float32}, modes1, modes2, modes3, in_channels, out_channels)
    weights4 = scale*rand(Complex{Float32}, modes1, modes2, modes3, in_channels, out_channels)
    return SpectralConv3d_fast{Complex{Float32}, 5}(weights1, weights2, weights3, weights4)
end

function compl_mul3d(x::AbstractArray{Complex{Float32}}, y::AbstractArray{Complex{Float32}})
    # complex multiplication
    # x in (modes1, modes2, modes3, input channels, batchsize)
    # y in (modes1, modes2, modes3, input channels, output channels)
    # output in (modes1,modes2,output channles,batchsize)
    x_per = permutedims(x,[5,4,1,2,3]) # batchsize*in_channels*modes1*modes2
    y_per = permutedims(y,[4,5,1,2,3]) # in_channels*out_channels*modes1*modes2
    x_resh = reshape(x_per,size(x_per,1),size(x_per,2),:) # batchsize*in_channels*(modes1*modes2*modes3)
    y_resh = reshape(y_per,size(y_per,1),size(y_per,2),:) # in_channels*out_channels*(modes1*modes2*modes3)
    out_resh = batched_mul(x_resh,y_resh) # batchsize*out_channels*(modes1*modes2*modes3)
    out_per = reshape(out_resh,size(out_resh,1),size(out_resh,2),size(x,1),size(x,2),size(x,3)) # batchsize*out_channels*modes1*modes2*modes3
    out = permutedims(out_per,[3,4,5,2,1])
    return out
end

function (L::SpectralConv3d_fast)(x::AbstractArray{Float32})
    # x in (size_x, size_y, time, channels, batchsize
    x_ft = rfft(x,[1,2,3])
    modes1 = size(L.weights1,1)
    modes2 = size(L.weights1,2)
    modes3 = size(L.weights1,3)
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

mutable struct SimpleBlock3d
    fc0::Conv
    conv0::SpectralConv3d_fast
    conv1::SpectralConv3d_fast
    conv2::SpectralConv3d_fast
    conv3::SpectralConv3d_fast
    w0::Conv
    w1::Conv
    w2::Conv
    w3::Conv
    bn0::BatchNorm
    bn1::BatchNorm
    bn2::BatchNorm
    bn3::BatchNorm
    fc1::Conv
    fc2::Conv
end

@Flux.functor SimpleBlock3d

function SimpleBlock3d(modes1::Integer, modes2::Integer, modes3::Integer, width::Integer)
    block = SimpleBlock3d(
        Conv((1, 1, 1), 4=>width),
        SpectralConv3d_fast(width, width, modes1, modes2, modes3),
        SpectralConv3d_fast(width, width, modes1, modes2, modes3),
        SpectralConv3d_fast(width, width, modes1, modes2, modes3),
        SpectralConv3d_fast(width, width, modes1, modes2, modes3),
        Conv((1, 1, 1), width=>width),
        Conv((1, 1, 1), width=>width),
        Conv((1, 1, 1), width=>width),
        Conv((1, 1, 1), width=>width),
        BatchNorm(width, identity; ϵ=1.0f-5, momentum=.1f0),
        BatchNorm(width, identity; ϵ=1.0f-5, momentum=.1f0),
        BatchNorm(width, identity; ϵ=1.0f-5, momentum=.1f0),
        BatchNorm(width, identity; ϵ=1.0f-5, momentum=.1f0),
        Conv((1, 1, 1), width=>128),
        Conv((1, 1, 1), 128=>1)
    )
    return block
end

function (B::SimpleBlock3d)(x::AbstractArray{Float32})
    x = B.fc0(x)
    x1 = B.conv0(x)
    x2 = B.w0(x)
    x = B.bn0(x1+x2)
    x = relu.(x)
    x1 = B.conv1(x)
    x2 = B.w1(x)
    x = B.bn1(x1+x2)
    x = relu.(x)
    x1 = B.conv2(x)
    x2 = B.w2(x)
    x = B.bn2(x1+x2)
    x = relu.(x)
    x1 = B.conv3(x)
    x2 = B.w3(x)
    x = B.bn3(x1+x2)
    x = B.fc1(x)
    x = relu.(x)
    x = B.fc2(x)
    return x
end

mutable struct Net3d
    conv1::SimpleBlock3d
end

@Flux.functor Net3d

function Net3d(modes::Integer, width::Integer)
    return Net3d(SimpleBlock3d(modes,modes,modes,width))
end

function (NN::Net3d)(x::AbstractArray{Float32})
    x = NN.conv1(x)
    x = dropdims(x,dims=4)
end


ntrain = 1000
ntest = 100

BSON.@load "2phasenet_100.bson" NN w batch_size Loss modes width learning_rate epochs gamma step_size

n = (64,64)
#d = (15f0,15f0) # dx, dy in m
d = (1f0/64, 1f0/64)

nt = 51
#dt = 20f0    # dt in day
dt = 1f0/nt

perm = matread("data/perm.mat")["perm"]
conc = matread("data/conc.mat")["conc"]

x_train_ = convert(Array{Float32},perm[:,:,1:ntrain])

y_train_ = convert(Array{Float32},conc[:,:,:,1:ntrain])

y_train_ = permutedims(y_train_,[2,3,1,4])

x_normalizer = UnitGaussianNormalizer(x_train_)
x_train_ = encode(x_normalizer,x_train_)

y_normalizer = UnitGaussianNormalizer(y_train_)

x = reshape(collect(range(d[1],stop=n[1]*d[1],length=n[1])), :, 1)
z = reshape(collect(range(d[2],stop=n[2]*d[2],length=n[2])), 1, :)

grid = zeros(Float32,n[1],n[2],2)
grid[:,:,1] = repeat(x',n[2])'
grid[:,:,2] = repeat(z,n[1])

x_train = zeros(Float32,n[1],n[2],nt,4,ntrain)

for i = 1:nt
    x_train[:,:,i,1,:] = deepcopy(x_train_)
    for j = 1:ntrain
        x_train[:,:,i,2,j] = grid[:,:,1]
        x_train[:,:,i,3,j] = grid[:,:,2]
        x_train[:,:,i,4,j] .= i*dt
    end
end

# value, x, y, t

compute_time = 0f0

x_train_new = zeros(Float32,1000,1000,51,4,1000)
for i = 1:51
    for j = 1:4
        for k  = 1:1000
            x_train_new[:,:,i,j,k] = imresize(x_train[:,:,i,j,k])
        end
    end
end
run_samples = 100
for i = 1:run_samples
    in_sample = x_train_new[:,:,:,:,i:i]
    start_time = time();
    @time out_sample = NN(in_sample)
    end_time = time();
    y_out = decode(y_normalizer,out_sample);
    run_time = end_time-start_time
    #println(run_time)
    global compute_time += run_time
end

println("Average time over ", run_samples, " samples = ", compute_time/run_samples, " seconds")