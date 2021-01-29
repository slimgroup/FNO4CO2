# author: Ziyi Yin
# This code is an implementation of fourier neural operators from Zongyi Li's repository

using PyPlot
using Flux, Random, FFTW, Zygote, NNlib
using MAT, Statistics, LinearAlgebra
using OMEinsum

include("utils.jl")

Random.seed!(3)

mutable struct SpectralConv2d
    weights1
    weights2
end

@Flux.functor SpectralConv2d

# Constructor
function SpectralConv2d(in_channels, out_channels, modes1, modes2)
    scale = (1f0 / (in_channels * out_channels))
    weights1 = scale*rand(Complex{Float32}, modes1, modes2, in_channels, out_channels)
    weights2 = scale*rand(Complex{Float32}, modes1, modes2, in_channels, out_channels)
    return SpectralConv2d(weights1, weights2)
end

function compl_mul2d(x::AbstractArray{Complex{Float32}}, y::AbstractArray{Complex{Float32}})
    # complex multiplication
    # x in (modes1, modes2, input channels, batchsize)
    # y in (modes1, modes2, input channels, output channels, 2)
    # output in (modes1,modes2 output)
    out = ein"ijml, ijmk -> ijkl"(x,y)
    return out
end

function (L::SpectralConv2d)(x::AbstractArray{Float32})
    # x in (size_x, size_y, channels, batchsize)
    x_ft = rfft(x,[2,1])
    modes1 = size(L.weights1,1)
    modes2 = size(L.weights1,2)
    out_ft = cat(compl_mul2d(x_ft[1:modes1, 1:modes2,:,:], L.weights1),
        zeros(Complex{Float32},size(x_ft,1)-2*modes1,size(x_ft,2)-2*modes2,size(x_ft,3),size(x_ft,4)),
        compl_mul2d(x_ft[end-modes1+1:end, 1:modes2,:,:], L.weights2),dims=(1,2))
    x = irfft(out_ft, size(x,2),[2,1])
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
        Conv((1, 1), 3=>width),
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
        Conv((1, 1), width=>128),
        Conv((1, 1), 128=>1)
    )
    return block
end

function (B::SimpleBlock2d)(x::AbstractArray{Float32})
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

mutable struct Net2d
    conv1
end

@Flux.functor Net2d

function Net2d(modes, width)
    return Net2d(SimpleBlock2d(modes,modes,width))
end

function (NN::Net2d)(x::AbstractArray{Float32})
    x = NN.conv1(x)
    x = dropdims(x,dims=3)
end


ntrain = 1000
ntest = 100

batch_size = 20
learning_rate = 1f-3

epochs = 500
step_size = 100
gamma = 5f-1

modes = 12
width = 32

r = 5
h = Int(((421 - 1)/r) + 1)
s = h

TRAIN = matread("data/piececonst_r421_N1024_smooth1.mat")
x_train_ = convert(Array{Float32},TRAIN["coeff"][1:ntrain,1:r:end,1:r:end][:,1:s,1:s])
y_train_ = convert(Array{Float32},TRAIN["sol"][1:ntrain,1:r:end,1:r:end][:,1:s,1:s])

TEST = matread("data/piececonst_r421_N1024_smooth2.mat")
x_test_ = convert(Array{Float32},TEST["coeff"][1:ntest,1:r:end,1:r:end][:,1:s,1:s])
y_test = convert(Array{Float32},TEST["sol"][1:ntest,1:r:end,1:r:end][:,1:s,1:s])

x_train_ = permutedims(x_train_,[2,3,1])
x_test_ = permutedims(x_test_,[2,3,1])
y_train_ = permutedims(y_train_,[2,3,1])
y_test = permutedims(y_test,[2,3,1])

x_normalizer = UnitGaussianNormalizer(x_train_)
x_train_ = encode(x_normalizer,x_train_)
x_test_ = encode(x_normalizer,x_test_)

y_normalizer = UnitGaussianNormalizer(y_train_)
y_train = encode(y_normalizer,y_train_)
#y_test = encode(y_normalizer,y_test_)

x = reshape(collect(range(0f0,stop=1f0,length=s)), :, 1)
z = reshape(collect(range(0f0,stop=1f0,length=s)), 1, :)

grid = zeros(Float32,s,s,2)
grid[:,:,1] = repeat(z,s)
grid[:,:,2] = repeat(x',s)'

x_train = zeros(Float32,s,s,3,ntrain)
x_train[:,:,1,:] = x_train_

for i = 1:ntrain
    x_train[:,:,2,i] = grid[:,:,1]
    x_train[:,:,3,i] = grid[:,:,2]
end

x_test = zeros(Float32,s,s,3,ntest)
x_test[:,:,1,:] = x_test_

for i = 1:ntest
    x_test[:,:,2,i] = grid[:,:,1]
    x_test[:,:,3,i] = grid[:,:,2]
end

train_loader = Flux.Data.DataLoader((x_train, y_train); batchsize = batch_size, shuffle = true)
test_loader = Flux.Data.DataLoader((x_test, y_test); batchsize = batch_size, shuffle = false)

NN = Net2d(modes, width) |> gpu

w = Flux.params(NN)
Flux.trainmode!(NN, true)
opt = Flux.Optimise.ADAMW(learning_rate, (0.9f0, 0.999f0), 1f-4)

Loss = zeros(Float32,epochs)
for ep = 1:epochs
    for (x,y) in train_loader
        grads = gradient(w) do
            out = decode(y_normalizer,NN(x))    |> gpu
            y_n = decode(y_normalizer,y)        |> gpu
            global loss = 1f0/(s-1)*Flux.mse(out,y_n)
            return loss
        end
        for p in w
            Flux.Optimise.update!(opt, p, grads[p])
        end
    end
    println(" Epoch: ", ep, " | Objective = ", loss)
    Loss[ep] = loss
end

figure();plot(Loss);title("History");xlabel("Epochs");ylabel("Loss")

Flux.testmode!(NN, true)

x_test_1 = x_test[:,:,:,1:1]
x_test_2 = x_test[:,:,:,2:2]
x_test_3 = x_test[:,:,:,3:3]

y_test_1 = y_test[:,:,1]
y_test_2 = y_test[:,:,2]
y_test_3 = y_test[:,:,3]

y_predict_1 = decode(y_normalizer,NN(x_test_1))[:,:,1]
y_predict_2 = decode(y_normalizer,NN(x_test_2))[:,:,1]
y_predict_3 = decode(y_normalizer,NN(x_test_3))[:,:,1]

figure(figsize=(9,9));
subplot(3,3,1);
title("sample 1")
imshow(x_test_1[:,:,1])
subplot(3,3,2);
title("sample 2")
imshow(x_test_2[:,:,1])
subplot(3,3,3);
title("sample 3")
imshow(x_test_3[:,:,1])
subplot(3,3,4);
title("predict 1")
imshow(y_predict_1)
subplot(3,3,5);
title("predict 2")
imshow(y_predict_2)
subplot(3,3,6);
title("predict 3")
imshow(y_predict_3)
subplot(3,3,7);
title("true 1")
imshow(y_test_1)
subplot(3,3,8);
title("true 2")
imshow(y_test_2)
subplot(3,3,9);
title("true 3")
imshow(y_test_3)

savefig("result/500ep.png")
