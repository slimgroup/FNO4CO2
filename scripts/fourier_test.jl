# author: Ziyi Yin, ziyi.yin@gatech.edu
# This script trains a Fourier Neural Operator which maps 2D permeability distribution to time-varying CO2 concentration snapshots.
# The PDE is in 2D while FNO requires 3D FFT

using Pkg
Pkg.activate("./")
using DrWatson
@quickactivate "FNO4CO2"

using FNO4CO2
using PyPlot
using JLD2
using Flux, Random, FFTW
using MAT, Statistics, LinearAlgebra
using CUDA
using InvertibleNetworks:ActNorm
using Profile
matplotlib.use("Agg")

Random.seed!(1234)

# Define raw data directory
mkpath(datadir("training-data"))
perm_path = datadir("training-data", "perm_gridspacing15.0.mat")
conc_path = datadir("training-data", "conc_gridspacing15.0.mat")

# Download the dataset into the data directory if it does not exist
if ~isfile(perm_path)
    run(`wget https://www.dropbox.com/s/o35wvnlnkca9r8k/'
        'perm_gridspacing15.0.mat -q -O $perm_path`)
end
if ~isfile(conc_path)
    run(`wget https://www.dropbox.com/s/mzi0xgr0z3l553a/'
        'conc_gridspacing15.0.mat -q -O $conc_path`)
end

# perm = matread(perm_path)["perm"];
# conc = matread(conc_path)["conc"];

nsamples = 2
dim = 64
nt = 51

perm = rand(Float32, dim, dim, nsamples)
conc = rand(Float32, nt, dim, dim, nsamples)

ntrain = 1
nvalid = 1

batch_size = 2
learning_rate = 1f-4

modes = 4
width = 20

n = (dim,dim)
#d = (15f0,15f0) # dx, dy in m
d = (1f0/dim, 1f0/dim)

s = 1

#dt = 20f0    # dt in day
dt = 1f0/(nt-1)

AN = ActNorm(ntrain)
AN.forward(reshape(perm[1:s:end,1:s:end,1:ntrain], n[1], n[2], 1, ntrain));

y_train = permutedims(conc[1:nt,1:s:end,1:s:end,1:ntrain],[2,3,1,4]);
y_valid = permutedims(conc[1:nt,1:s:end,1:s:end,ntrain+1:ntrain+nvalid],[2,3,1,4]);

grid = gen_grid(n, d, nt, dt)

x_train = perm_to_tensor(perm[1:s:end,1:s:end,1:ntrain],grid,AN);
x_valid = perm_to_tensor(perm[1:s:end,1:s:end,ntrain+1:ntrain+nvalid],grid,AN);

# value, x, y, t

NN = Net3d(modes, width)
gpu_flag && (global NN = NN |> gpu)

Flux.trainmode!(NN, true)
w = Flux.params(NN)

Base.flush(Base.stdout)

Flux.trainmode!(NN, true)

x = x_train[:, :, :, :, 1:1]
y = y_train[:, :, :, 1:1]

# if gpu_flag
#     x = x |> gpu
#     y = y |> gpu
# end

@time NN(x)
# println("SECOND CALL")
# Profile.clear_malloc_data()
@time NN(x)
@time NN(x)

@time grads = gradient(w) do
    global loss = norm(relu01(NN(x))-y)/norm(y)
    return loss
end
@time grads = gradient(w) do
    global loss = norm(relu01(NN(x))-y)/norm(y)
    return loss
end
@time grads = gradient(w) do
    global loss = norm(relu01(NN(x))-y)/norm(y)
    return loss
end
