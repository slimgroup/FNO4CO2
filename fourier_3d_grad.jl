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

grad_iterations = 50

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

fix_input = randn(Float32, nx, ny)
temp1 = decode(y_normalizer,NN(perm_to_tensor(fix_input,nt,grid,dt)))

function f(x_inv)
    println("evaluate f")
    @time begin
        out = decode(y_normalizer,NN(perm_to_tensor(x_inv,nt,grid,dt)))
        loss = Flux.mse(out,y_test_1; agg=sum)
    end
    return loss
end

function g!(gvec, x_inv)
    println("evaluate g")
    p = params(x_inv)
    @time grads = gradient(p) do
        out = decode(y_normalizer,NN(perm_to_tensor(x_inv,nt,grid,dt)))
        global loss = Flux.mse(out,y_test_1; agg=sum)
        return loss
    end
    copyto!(gvec, grads.grads[x_inv])
end

function fg!(gvec, x_inv)
    println("evaluate f and g")
    p = params(x_inv)
    @time grads = gradient(p) do
        out = decode(y_normalizer,NN(perm_to_tensor(x_inv,nt,grid,dt)))
        global loss = Flux.mse(out,y_test_1; agg=sum)
        return loss
    end
    copyto!(gvec, grads.grads[x_inv])
    return loss
end

function gdoptimize(f, g!, fg!, x0::AbstractArray{T}, linesearch;
                    maxiter::Int = 10000,
                    g_rtol::T = sqrt(eps(T)), g_atol::T = eps(T), init_α::T=T(1)) where T <: Number
    x = copy(x0)::AbstractArray{T}
    gvec = similar(x)::AbstractArray{T}
    fx = fg!(gvec, x)::T
    println("Initial loss = $fx")
    gnorm = norm(gvec)::T
    gtol = max(g_rtol*gnorm, g_atol)::T

    # Univariate line search functions
    ϕ(α) = f(x .+ α.*s)::T
    function dϕ(α::T)
        g!(gvec, x .+ α.*s)
        return dot(gvec, s)::T
    end
    function ϕdϕ(α::T)
        phi = fg!(gvec, x .+ α.*s)::T
        dphi = dot(gvec, s)::T
        return (phi, dphi)::Tuple{T,T}
    end

    s = similar(gvec)::AbstractArray{T} # Step direction

    iter = 0
    Loss = zeros(Float32, maxiter)
    while iter < maxiter && gnorm > gtol
        iter += 1
        s .= -gvec::AbstractArray{T}

        dϕ_0 = dot(s, gvec)::T
        α, fx = linesearch(ϕ, dϕ, ϕdϕ, init_α, fx, dϕ_0)

        @. x = x + α*s::AbstractArray{T}
        g!(gvec, x)
        gnorm = norm(gvec)::T
        Loss[iter] = fx
        println("iteration $iter, loss = $fx, step length α=$α")

        init_α = 1f1 * α
    end

    return (Loss[1:iter], x, iter)
end

x0 = zeros(Float32, nx, ny)

ls = BackTracking(c_1=1f-4,iterations=1000,maxstep=Inf32,order=3,ρ_hi=5f-1,ρ_lo=1f-1)
Grad_Loss, x_inv, numiter = gdoptimize(f, g!, fg!, x0, ls; maxiter=grad_iterations)

temp2 = decode(y_normalizer,NN(perm_to_tensor(fix_input,nt,grid,dt)))
@assert isapprox(temp1, temp2) # test if network is in test mode (i.e. doesnt' change)

x_out = decode(x_normalizer,reshape(x_inv,nx,ny,1))[:,:,1]

figure();plot(Grad_Loss);title("Loss history");xlabel("iterations");ylabel("loss");

figure();
subplot(1,2,1);
imshow(x_out,vmin=20,vmax=120);title("inversion by NN, $grad_iterations iter");
subplot(1,2,2);
imshow(decode(x_normalizer,x_test_1)[:,:,1,1,1],vmin=20,vmax=120);title("GT permeability");

savefig("result/graddescent.png")
