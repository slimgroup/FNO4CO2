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
using LineSearches
using Optim: optimize, LBFGS, Options, only_fg!

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
        x_train[:,:,i,4,j] .= (i-1)*dt
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

grad_iterations = 50

nv = nt
survey_indices = Int.(round.(range(1, stop=nt, length=nv)))

λ = 1f0 # 2 norm regularization

#x = encode(x_normalizer,20f0*ones(Float32,nx,ny))[:,:,1]
x = zeros(Float32, nx, ny)
x_init = decode(x_normalizer,reshape(x,nx,ny,1))[:,:,1]

function f(z)
    println("evaluate f")
    out = decode(y_normalizer,NN(perm_to_tensor(z,nt,grid,dt)))
    misfit = 0.5f0 * (nt-1)/(nv-1) * norm(out[survey_indices,:,:]-y_test_1[survey_indices,:,:])^2f0
    prior = 0.5f0 * λ * norm(z)^2f0
    loss = misfit + prior
    @show loss, misfit, prior
    println("fval is = ", loss)
    return loss
end

function fg!(fval,g,z)
    println("evaluate f and g")
    p = params(z)
    @time grads = gradient(p) do
        out = decode(y_normalizer,NN(perm_to_tensor(z,nt,grid,dt)))
        misfit = 0.5f0 * (nt-1)/(nv-1) * norm(out[survey_indices,:,:]-y_test_1[survey_indices,:,:])^2f0
        prior = 0.5f0 * λ * norm(z)^2f0
        global loss = misfit + prior
        @show loss, misfit, prior
        return loss
    end
    println("fval is = ", loss)
    copyto!(g, grads.grads[z])
    return loss
end

function callb(os)
    z = os.metadata["x"] #get current optimization variable (latent z)
    fval[curr_iter] = f(z)
    println("iter: "*string(curr_iter)*" f_val: "*string(fval[curr_iter]))
    flush(stdout)
    axloss.plot(fval[1:curr_iter])
    ax.imshow(decode(z), vmin=20, vmax=120)
    global curr_iter += 1

    #if you want the callback to stop the procedure under some conditions, return true
    return false
end

std_ = x_normalizer.std_[:,:,1]
eps_ = x_normalizer.eps_
mean_ = x_normalizer.mean_[:,:,1]
curr_iter = 1
_, ax = subplots(nrows=1, ncols=1, figsize=(20,12))
_, axloss = subplots(nrows=1, ncols=1, figsize=(20,12))

lbfgs_iters = 50
fval = zeros(Float32, lbfgs_iters+1)
res = optimize(only_fg!(fg!), x, LBFGS(), Options(iterations=lbfgs_iters, extended_trace=true, callback=callb))

x_true = decode(x_normalizer,x_test_1[:,:,1:1,1,1])[:,:,1]

figure(figsize=(20,12));
subplot(1,3,1)
imshow(x_init,vmin=20,vmax=120);title("initial permeability");
subplot(1,3,2);
imshow(decode(res.minimizer),vmin=20,vmax=120);title("inversion by NN, $(grad_iterations) iter");
subplot(1,3,3);
imshow(x_true,vmin=20,vmax=120);title("GT permeability");
