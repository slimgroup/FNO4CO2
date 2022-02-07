# author: Ziyi (Francis) Yin
# This script trains a Fourier Neural Operator which maps 2D permeability distribution to time-varying CO2 concentration snapshots.
# The PDE is in 2D while FNO requires 3D FFT

using DrWatson
@quickactivate "FNO"
import Pkg; Pkg.instantiate()

using PyPlot
using BSON
using Flux, Random, FFTW, Zygote, NNlib
using MAT, Statistics, LinearAlgebra
using CUDA
using ProgressMeter

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

include("utils.jl")
include("fno3dstruct.jl")

Random.seed!(3)

ntrain = 1000
ntest = 100

batch_size = 10
learning_rate = 1f-4

epochs = 200
step_size = 100
gamma = 5f-1

modes = 4
width = 20

n = (64,64)
#d = (15f0,15f0) # dx, dy in m
d = (1f0/64, 1f0/64)

s = 4

nt = 51
#dt = 20f0    # dt in day
dt = 1f0/nt

# Define raw data directory
mkpath(datadir("data"))
perm_path = datadir("data", "perm.mat")
conc_path = datadir("data", "conc.mat")

# Download the dataset into the data directory if it does not exist
if isfile(perm_path) == false
    run(`wget https://www.dropbox.com/s/xzicjq9fnessdif/'
        'perm.mat -q -O $perm_path`)
end
if isfile(conc_path) == false
    run(`wget https://www.dropbox.com/s/s7ph9gf2xwlu5mb/'
        'conc.mat -q -O $conc_path`)
end

perm = matread(perm_path)["perm"]
conc = matread(conc_path)["conc"]

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
        x_train[:,:,i,4,j] .= (i-1)*dt
    end

    for k = 1:ntest
        x_test[:,:,i,2,k] = grid[:,:,1]
        x_test[:,:,i,3,k] = grid[:,:,2]
        x_test[:,:,i,4,k] .= (i-1)*dt
    end
end

# value, x, y, t

train_loader = Flux.Data.DataLoader((x_train, y_train); batchsize = batch_size, shuffle = true)
test_loader = Flux.Data.DataLoader((x_test, y_test); batchsize = batch_size, shuffle = false)

if gpu_flag
    y_normalizer.mean_ = y_normalizer.mean_ |> gpu
    y_normalizer.std_ = y_normalizer.std_   |> gpu
    y_normalizer.eps_ = y_normalizer.eps_   |> gpu
    NN = Net3d(modes, width) |> gpu
else
    NN = Net3d(modes, width)
end

w = Flux.params(NN)
Flux.trainmode!(NN, true)
opt = Flux.Optimise.ADAMW(learning_rate, (0.9f0, 0.999f0), 1f-4)

Loss = zeros(Float32,epochs*Int(ntrain/batch_size))

prog = Progress(ntrain * epochs)

iter = 0
for ep = 1:epochs
    Base.flush(Base.stdout)
    for (x,y) in train_loader
        global iter = iter + 1
        grads = gradient(w) do
            if gpu_flag
                x = x |> gpu
                y = y |> gpu
            end
            out = decode(y_normalizer,NN(x))
            y_n = decode(y_normalizer,y)
            global loss = Flux.mse(out,y_n;agg=sum)
            return loss
        end
        Loss[iter] = loss
        for p in w
            Flux.Optimise.update!(opt, p, grads[p])
        end
        ProgressMeter.next!(prog; showvalues = [(:loss, loss), (:epoch, ep)])
    end
end

if gpu_flag
    NN = NN |> cpu
    w = convert.(Array,w) |> cpu
end

# Define result directory
mkpath(datadir("TrainedNet"))
BSON.@save "data/TrainedNet/2phasenet_$epochs.bson" NN w batch_size Loss modes width learning_rate epochs gamma step_size s n d nt dt
