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
Flux.testmode!(NN.conv1.bn0)
Flux.testmode!(NN.conv1.bn1)
Flux.testmode!(NN.conv1.bn2)
Flux.testmode!(NN.conv1.bn3)

nx, ny = n
dx, dy = d

x_test_1 = x_test[:,:,:,:,1:1]
y_test_1 = y_test[:,:,:,1:1]

x_inv = zeros(Float32,nx,ny)

p =  params(x_inv)

grad_iterations = 60
grad_steplen = 3f-2

function perm_to_tensor(x_perm,nt,grid,dt)
    # input nx*ny, output nx*ny*nt*4*1
    nx, ny = size(x_perm)
    x1 = reshape(x_perm,nx,ny,1,1,1)
    x2 = cat([x1 for i = 1:nt]...,dims=3)
    grid_1 = cat([reshape(grid[:,:,1],nx,ny,1,1,1) for i = 1:nt]...,dims=3)
    grid_2 = cat([reshape(grid[:,:,2],nx,ny,1,1,1) for i = 1:nt]...,dims=3)
    grid_t = cat([i*dt*ones(Float32,nx,ny,1,1,1) for i = 1:nt]...,dims=3)
    x_out = cat(x2,grid_1,grid_2,grid_t,dims=4)
    return x_out
end

opt = Flux.Optimise.ADAMW(grad_steplen, (0.9f0, 0.999f0), 1f-4)

fix_input = randn(Float32, nx, ny)
temp1 = decode(y_normalizer,NN(perm_to_tensor(fix_input,nt,grid,dt)))

figure();

Grad_Loss = zeros(Float32,grad_iterations)
for iter = 1:grad_iterations
    Base.flush(Base.stdout)
    @time grads = gradient(p) do
        out = decode(y_normalizer,NN(perm_to_tensor(x_inv,nt,grid,dt)))
        global loss = Flux.mse(out,y_test_1; agg=sum)
        return loss
    end
    Grad_Loss[iter] = loss
    println("loss at iteration ", iter, " = $loss")
    for w in p
        Flux.Optimise.update!(opt, w, grads[w])
    end
    imshow(decode(x_normalizer,reshape(x_inv,nx,ny,1))[:,:,1], vmin=20, vmax=120);
    title("inverted permeability after iter $iter")
end

temp2 = decode(y_normalizer,NN(perm_to_tensor(fix_input,nt,grid,dt)))
@assert temp1 == temp2 # test if network is in test mode (i.e. doesnt' change)

x_out = decode(x_normalizer,reshape(x_inv,nx,ny,1))[:,:,1]

figure();plot(Grad_Loss);title("ADAM history");xlabel("iterations");ylabel("loss");
savefig("result/inv_his.png")

figure();
subplot(1,2,1);
imshow(x_out,vmin=20,vmax=120);title("inversion by NN, $grad_iterations iter");
subplot(1,2,2);
imshow(decode(x_normalizer,x_test_1)[:,:,1,1,1],vmin=20,vmax=120);title("GT permeability");
savefig("result/graddescent.png")