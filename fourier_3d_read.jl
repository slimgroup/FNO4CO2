# author: Ziyi Yin
# This code is an implementation of fourier neural operators from Zongyi Li's repository

using DrWatson
@quickactivate "FNO"
import Pkg; Pkg.instantiate()

using PyPlot
using BSON
using Flux, Random, FFTW, Zygote, NNlib
using MAT, Statistics, LinearAlgebra
using CUDA
using ProgressMeter, JLD2

CUDA.culiteral_pow(::typeof(^), a::Complex{Float32}, b::Val{2}) = real(conj(a)*a)
CUDA.sqrt(a::Complex) = cu(sqrt(a))
Base.broadcasted(::typeof(sqrt), a::Base.Broadcast.Broadcasted) = Base.broadcast(sqrt, Base.materialize(a))

include("utils.jl")
include("fno3dstruct.jl")
Random.seed!(3)

ntrain = 1000
ntest = 100

BSON.@load "data/TrainedNet/2phasenet_200.bson" NN w batch_size Loss modes width learning_rate epochs gamma step_size

n = (64,64)
#d = (15f0,15f0) # dx, dy in m
d = (1f0/64, 1f0/64)

nt = 51
#dt = 20f0    # dt in day
dt = 1f0/nt

perm = matread("data/data/perm.mat")["perm"]
conc = matread("data/data/conc.mat")["conc"]

s = 4

x_train_ = convert(Array{Float32},perm[1:s:end,1:s:end,1:ntrain])
x_test_ = convert(Array{Float32},perm[1:s:end,1:s:end,end-ntest+1:end])

y_train_ = convert(Array{Float32},conc[:,1:s:end,1:s:end,1:ntrain])
y_test_ = convert(Array{Float32},conc[:,1:s:end,1:s:end,end-ntest+1:end])

y_train_ = permutedims(y_train_,[2,3,1,4])
y_test = permutedims(y_test_,[2,3,1,4])

x_normalizer = UnitGaussianNormalizer(x_train_)
x_train_ = encode(x_normalizer,x_train_)
x_test_ = encode(x_normalizer,x_test_)

y_normalizer = UnitGaussianNormalizer(y_train_)
y_train = encode(y_normalizer,y_train_)

x = reshape(collect(range(d[1],stop=n[1]*d[1],length=n[1])), :, 1)
z = reshape(collect(range(d[2],stop=n[2]*d[2],length=n[2])), 1, :)

grid = zeros(Float32,n[1],n[2],2)
grid[:,:,1] = repeat(x',n[2])'
grid[:,:,2] = repeat(z,n[1])

x_train = zeros(Float32,n[1],n[2],nt,4,ntrain)
x_test = zeros(Float32,n[1],n[2],nt,4,ntest)

for i = 1:nt
    x_train[:,:,i,1,:] = deepcopy(x_train_)
    x_test[:,:,i,1,:] = deepcopy(x_test_)
    for j = 1:ntrain
        x_train[:,:,i,2,j] = grid[:,:,1]
        x_train[:,:,i,3,j] = grid[:,:,2]
        x_train[:,:,i,4,j] .= i*dt
    end

    for k = 1:ntest
        x_test[:,:,i,2,k] = grid[:,:,1]
        x_test[:,:,i,3,k] = grid[:,:,2]
        x_test[:,:,i,4,k] .= (i-1)*dt
    end
end

# value, x, y, t

Flux.testmode!(NN, true)
opt = Flux.Optimise.ADAMW(learning_rate, (0.9f0, 0.999f0), 1f-4)

figure();plot(Loss);xlabel("iterations");ylabel("Loss");title("Loss history on training batch")

savefig("result/loss_$epochs.png")

x_test_1 = x_test[:,:,:,:,1:1]
x_test_2 = x_test[:,:,:,:,2:2]
x_test_3 = x_test[:,:,:,:,3:3]

x_train_1 = x_train[:,:,:,:,1:1]
x_train_2 = x_train[:,:,:,:,2:2]
x_train_3 = x_train[:,:,:,:,3:3]

y_test_1 = y_test[:,:,:,1]
y_test_2 = y_test[:,:,:,2]
y_test_3 = y_test[:,:,:,3]

y_train_1 = decode(y_normalizer,y_train)[:,:,:,1]
y_train_2 = decode(y_normalizer,y_train)[:,:,:,2]
y_train_3 = decode(y_normalizer,y_train)[:,:,:,3]

y_predict_1 = decode(y_normalizer,NN(x_test_1))[:,:,:,1]
y_predict_2 = decode(y_normalizer,NN(x_test_2))[:,:,:,1]
y_predict_3 = decode(y_normalizer,NN(x_test_3))[:,:,:,1]

y_fit_1 = decode(y_normalizer,NN(x_train_1))[:,:,:,1]
y_fit_2 = decode(y_normalizer,NN(x_train_2))[:,:,:,1]
y_fit_3 = decode(y_normalizer,NN(x_train_3))[:,:,:,1]

# fit on training

figure(figsize=(15,15));
for i = 1:9
    subplot(4,9,i);
    imshow(y_fit_1[:,:,6*i-5],vmin=0,vmax=1);
end
for i = 1:9
    subplot(4,9,i+9);
    imshow(y_train_1[:,:,6*i-5],vmin=0,vmax=1);
end
for i = 1:9
    subplot(4,9,i+18);
    imshow(20*(y_train_1[:,:,6*i-5]-y_fit_1[:,:,6*i-5]),vmin=0,vmax=1);
end
subplot(4,9,28);
imshow(decode(x_normalizer,x_train_1)[:,:,1,1,1],vmin=20,vmax=120)
suptitle("Training sample 1: 1st row predict; 2nd row grond truth; 3rd row 20*diff; last row permeability")

savefig("result/2phase_trainsample1.png")

figure(figsize=(15,15));
for i = 1:9
    subplot(4,9,i);
    imshow(y_fit_2[:,:,6*i-5],vmin=0,vmax=1);
end
for i = 1:9
    subplot(4,9,i+9);
    imshow(y_train_2[:,:,6*i-5],vmin=0,vmax=1);
end
for i = 1:9
    subplot(4,9,i+18);
    imshow(20*(y_train_2[:,:,6*i-5]-y_fit_2[:,:,6*i-5]),vmin=0,vmax=1);
end
subplot(4,9,28);
imshow(decode(x_normalizer,x_train_2)[:,:,1,1,1],vmin=20,vmax=120)
suptitle("Training sample 2: 1st row predict; 2nd row grond truth; 3rd row 20*diff; last row permeability")

savefig("result/2phase_trainsample2.png")

figure(figsize=(15,15));
for i = 1:9
    subplot(4,9,i);
    imshow(y_fit_3[:,:,6*i-5],vmin=0,vmax=1);
end
for i = 1:9
    subplot(4,9,i+9);
    imshow(y_train_3[:,:,6*i-5],vmin=0,vmax=1);
end
for i = 1:9
    subplot(4,9,i+18);
    imshow(20*(y_train_3[:,:,6*i-5]-y_fit_3[:,:,6*i-5]),vmin=0,vmax=1);
end
subplot(4,9,28);
imshow(decode(x_normalizer,x_train_3)[:,:,1,1,1],vmin=20,vmax=120)
suptitle("Training sample 3: 1st row predict; 2nd row grond truth; 3rd row 20*diff; last row permeability")

savefig("result/2phase_trainsample3.png")

# test on test set

figure(figsize=(20,7));

for i = 1:9
    subplot(4,9,i);
    imshow(y_predict_1[:,:,6*i-5],vmin=0,vmax=1);
end
for i = 1:9
    subplot(4,9,i+9);
    imshow(y_test_1[:,:,6*i-5],vmin=0,vmax=1);
end
for i = 1:9
    subplot(4,9,i+18);
    imshow(10*(y_test_1[:,:,6*i-5]-y_predict_1[:,:,6*i-5]),vmin=0,vmax=1);
end
subplot(4,9,28);
imshow(decode(x_normalizer,x_test_1)[:,:,1,1,1],vmin=20,vmax=120)
suptitle("1st row predict; 2nd row grond truth; 3rd row 10*diff; last row permeability")

savefig("result/2phase_testsample1.png")


figure(figsize=(23,5));

for i = 1:9
    subplot(3,9,i);
    imshow(y_predict_1[:,:,6*i-5],vmin=0,vmax=1);
end
for i = 1:9
    subplot(3,9,i+9);
    imshow(y_test_1[:,:,6*i-5],vmin=0,vmax=1);
end
for i = 1:9
    subplot(3,9,i+18);
    imshow(10*(y_test_1[:,:,6*i-5]-y_predict_1[:,:,6*i-5]),vmin=0,vmax=1);
end
#suptitle("1st row predict; 2nd row grond truth; 3rd row 10*diff")
savefig("result/2phase_testsample1.png", bbox_inches="tight", dpi=150)

figure()
imshow(perm[:,:,1001],vmin=20,vmax=120);
#title("permeability");
savefig("result/2phase_perm1.png", bbox_inches="tight", dpi=150)


figure(figsize=(15,15));

for i = 1:9
    subplot(4,9,i);
    imshow(y_predict_2[:,:,6*i-5],vmin=0,vmax=1);
end
for i = 1:9
    subplot(4,9,i+9);
    imshow(y_test_2[:,:,6*i-5],vmin=0,vmax=1);
end
for i = 1:9
    subplot(4,9,i+18);
    imshow(20*(y_test_2[:,:,6*i-5]-y_predict_2[:,:,6*i-5]),vmin=0,vmax=1);
end
subplot(4,9,28);
imshow(decode(x_normalizer,x_test_2)[:,:,1,1,1],vmin=20,vmax=120)
suptitle("Test sample 2: 1st row predict; 2nd row grond truth; 3rd row 20*diff; last row permeability")

savefig("result/2phase_testsample2.png")

figure(figsize=(15,15));

for i = 1:9
    subplot(4,9,i);
    imshow(y_predict_3[:,:,6*i-5],vmin=0,vmax=1);
end
for i = 1:9
    subplot(4,9,i+9);
    imshow(y_test_3[:,:,6*i-5],vmin=0,vmax=1);
end
for i = 1:9
    subplot(4,9,i+18);
    imshow(20*(y_test_3[:,:,6*i-5]-y_predict_3[:,:,6*i-5]),vmin=0,vmax=1);
end
subplot(4,9,28);
imshow(decode(x_normalizer,x_test_3)[:,:,1,1,1],vmin=20,vmax=120)
suptitle("Test sample 3: 1st row predict; 2nd row grond truth; 3rd row 20*diff; last row permeability")

savefig("result/2phase_testsample3.png")