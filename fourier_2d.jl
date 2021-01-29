# author: Ziyi Yin
# This code is an implementation of fourier neural operators from Zongyi Li's repository

using PyPlot
using Flux, Random, FFTW, Zygote, NNlib
using MAT, Statistics, LinearAlgebra
using CUDA

CUDA.culiteral_pow(::typeof(^), a::Complex{Float32}, b::Val{2}) = real(conj(a)*a)
CUDA.sqrt(a::Complex) = cu(sqrt(a))
Base.broadcasted(::typeof(sqrt), a::Base.Broadcast.Broadcasted) = Base.broadcast(sqrt, Base.materialize(a))

include("utils.jl")

Random.seed!(3)

mutable struct SpectralConv2d{T,N}
    weights1::AbstractArray{T,N}
    weights2::AbstractArray{T,N}
end

@Flux.functor SpectralConv2d

# Constructor
function SpectralConv2d(in_channels::Integer, out_channels::Integer, modes1::Integer, modes2::Integer)
    scale = (1f0 / (in_channels * out_channels))
    weights1 = scale*rand(Complex{Float32}, modes1, modes2, in_channels, out_channels) |> gpu
    weights2 = scale*rand(Complex{Float32}, modes1, modes2, in_channels, out_channels) |> gpu
    return SpectralConv2d{Complex{Float32}, 4}(weights1, weights2)
end

function compl_mul2d(x::AbstractArray{Complex{Float32}}, y::AbstractArray{Complex{Float32}})
    # complex multiplication
    # x in (modes1, modes2, input channels, batchsize)
    # y in (modes1, modes2, input channels, output channels)
    # output in (modes1,modes2,output channles,batchsize)
    x_per = permutedims(x,[4,3,1,2]) # batchsize*in_channels*modes1*modes2
    y_per = permutedims(y,[3,4,1,2]) # in_channels*out_channels*modes1*modes2
    x_resh = reshape(x_per,size(x_per,1),size(x_per,2),:) # batchsize*in_channels*(modes1*modes2)
    y_resh = reshape(y_per,size(y_per,1),size(y_per,2),:) # in_channels*out_channels*(modes1*modes2)
    out_resh = batched_mul(x_resh,y_resh) # batchsize*out_channels*(modes1*modes2)
    out_per = reshape(out_resh,size(out_resh,1),size(out_resh,2),size(x,1),size(x,2)) # batchsize*out_channels*modes1*modes2
    out = permutedims(out_per,[3,4,2,1])
    return out
end

function (L::SpectralConv2d)(x::AbstractArray{Float32})
    # x in (size_x, size_y, channels, batchsize)
    x_ft = rfft(x,[1,2])
    modes1 = size(L.weights1,1)
    modes2 = size(L.weights1,2)
    zs = 0f0im .* view(x_ft, 1:size(x_ft, 1)-2*modes1, 1:size(x_ft, 2)-2*modes2, :, :)
    out_ft = cat(compl_mul2d(x_ft[1:modes1, 1:modes2,:,:], L.weights1), zs,
                 compl_mul2d(x_ft[end-modes1+1:end, 1:modes2,:,:], L.weights2),dims=(1,2))
    x = irfft(out_ft, size(x,1),[1,2])
end

mutable struct SimpleBlock2d
    fc0::Conv
    conv0::SpectralConv2d
    conv1::SpectralConv2d
    conv2::SpectralConv2d
    conv3::SpectralConv2d
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

@Flux.functor SimpleBlock2d

function SimpleBlock2d(modes1::Integer, modes2::Integer, width::Integer)
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

function Net2d(modes::Integer, width::Integer)
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

epochs = 200
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

y_normalizer.mean_ = y_normalizer.mean_ |> gpu
y_normalizer.std_ = y_normalizer.std_   |> gpu
y_normalizer.eps_ = y_normalizer.eps_   |> gpu

NN = Net2d(modes, width) |> gpu

w = Flux.params(NN)
Flux.trainmode!(NN, true)
opt = Flux.Optimise.ADAMW(learning_rate, (0.9f0, 0.999f0), 1f-4)

Loss = zeros(Float32,epochs)
for ep = 1:epochs
    for (x,y) in train_loader
        grads = gradient(w) do
            x = x |> gpu
            y = y |> gpu
            out = decode(y_normalizer,NN(x))
            y_n = decode(y_normalizer,y)
            global loss = Flux.mse(out,y_n;agg=sum)
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
