# author: Ziyi Yin
# This code is an implementation of fourier neural operators from Zongyi Li's repository

using PyPlot
using Flux, Random, FFTW, Zygote, NNlib
using Einsum
using MAT, Statistics

Random.seed!(3)

mutable struct SpectralConv2d
    weights1
    weights2
end

@Flux.functor SpectralConv2d

# Constructor
function SpectralConv2d(in_channels, out_channels, modes1, modes2)
    scale = (1 / (in_channels * out_channels))
    weights1 = scale*rand(Float32, modes1, modes2, in_channels, out_channels, 2)
    weights2 = scale*rand(Float32, modes1, modes2, in_channels, out_channels, 2)
    return SpectralConv2d(weights1, weights2)
end

function compl_mul2d(x, y)
    # complex multiplication
    # x in (modes1, modes2, input channels, batchsize)
    # y in (modes1, modes2, input channels, output channels, 2)
    # output in (modes1,modes2 output)
    f_einsum(A,B) = @einsum C[i,j,k,l] := A[i,j,m,l] * B[i,j,m,k]
    return f_einsum(x,y[:,:,:,:,1]+im*y[:,:,:,:,2])
end

function (L::SpectralConv2d)(x::AbstractArray)
    # x in (size_x, size_y, channels, batchsize)
    x_ft = rfft(x,[2,1])
    out_ft = zeros(eltype(x_ft),size(x_ft))
    modes1 = size(L.weights1,1)
    modes2 = size(L.weights1,2)
    out_ft[1:modes1,1:modes2,:,:] = compl_mul2d(x_ft[1:modes1, 1:modes2,:,:], L.weights1)
    out_ft[end-modes1+1:end,1:modes2,:,:] = compl_mul2d(x_ft[end-modes1+1:end, 1:modes2,:,:], L.weights2)
    x = irfft(out_ft, size(x,1),[2,1])
end


mutable struct SimpleBlock2d
    fc0
    conv0
    conv1
    conv2
    conv3
    w0
    w1
    w2
    w3
    bn0
    bn1
    bn2
    bn3
    fc1
    fc2
end

@Flux.functor SimpleBlock2d

function SimpleBlock2d(modes1, modes2, width)
    block = SimpleBlock2d(
        Dense(3, width),
        SpectralConv2d(width, width, modes1, modes2),
        SpectralConv2d(width, width, modes1, modes2),
        SpectralConv2d(width, width, modes1, modes2),
        SpectralConv2d(width, width, modes1, modes2),
        Conv((1, 1), width=>width),
        Conv((1, 1), width=>width),
        Conv((1, 1), width=>width),
        Conv((1, 1), width=>width),
        BatchNorm(width, identity; ϵ=1.0f-5, momentum=.1f0),
        BatchNorm(width, identity; ϵ=1.0f-5, momentum=.1f0),
        BatchNorm(width, identity; ϵ=1.0f-5, momentum=.1f0),
        BatchNorm(width, identity; ϵ=1.0f-5, momentum=.1f0),
        Dense(width, 128),
        Dense(128,1)
    )
    return block
end

function (B::SimpleBlock2d)(x::AbstractArray)
    size_x = size(x)
    x = reshape(B.fc0(reshape(x,3,:)),size_x[1],size_x[2],:,size_x[end])
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
    
    x = reshape(B.fc1(reshape(x,width,:)),size_x[1],size_x[2],:,size_x[end])
    x = relu.(x)
    x = reshape(B.fc2(reshape(x,128,:)),size_x[1],size_x[2],:,size_x[end])
    return x
end

mutable struct Net2d
    conv1
end

@Flux.functor Net2d

function Net2d(modes, width)
    return Net2d(SimpleBlock2d(modes,modes,width))
end

function (NN::Net2d)(x::AbstractArray)
    x = NN.conv1(x)
    x = dropdims(x,dims=3)
end


ntrain = 1000
ntest = 100

batch_size = 20
learning_rate = 0.001

epochs = 500
step_size = 100
gamma = 0.5

modes = 12
width = 32

r = 5
h = Int(((421 - 1)/r) + 1)
s = h

x = randn(s, h, 3, batch_size)

NN = Net2d(modes, width)

y = NN(x)

TRAIN = matread("data/piececonst_r421_N1024_smooth1.mat")
x_train = TRAIN["coeff"][1:ntrain,1:r:end,1:r:end][:,1:s,1:s];
y_train = TRAIN["sol"][1:ntrain,1:r:end,1:r:end][:,1:s,1:s];

TEST = matread("data/piececonst_r421_N1024_smooth2.mat")
x_test = TRAIN["coeff"][1:ntest,1:r:end,1:r:end][:,1:s,1:s];
y_test = TRAIN["sol"][1:ntest,1:r:end,1:r:end][:,1:s,1:s];
