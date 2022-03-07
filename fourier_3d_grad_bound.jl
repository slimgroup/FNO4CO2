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

nv = nt
survey_indices = Int.(round.(range(1, stop=nt, length=nv)))

function f(x_inv)
    println("evaluate f")
    @time begin
        out = decode(y_normalizer,NN(perm_to_tensor(x_inv,nt,grid,dt)))
        loss = 0.5f0 * norm(out[survey_indices,:,:]-y_test_1[survey_indices,:,:])^2f0
    end
    return loss
end

function g!(gvec, x_inv)
    println("evaluate g")
    p = params(x_inv)
    @time grads = gradient(p) do
        out = decode(y_normalizer,NN(perm_to_tensor(x_inv,nt,grid,dt)))
        global loss = 0.5f0 * norm(out[survey_indices,:,:]-y_test_1[survey_indices,:,:])^2f0
        return loss
    end
    copyto!(gvec, grads.grads[x_inv])
end

function fg!(gvec, x_inv)
    println("evaluate f and g")
    p = params(x_inv)
    @time grads = gradient(p) do
        out = decode(y_normalizer,NN(perm_to_tensor(x_inv,nt,grid,dt)))
        global loss = 0.5f0 * norm(out[survey_indices,:,:]-y_test_1[survey_indices,:,:])^2f0
        return loss
    end
    copyto!(gvec, grads.grads[x_inv])
    return loss
end

function prj(x, vmin, vmax)
    x_perm = decode(x_normalizer,reshape(x,nx,ny,1))[:,:,1]
    x_perm1 = min.(max.(x_perm,vmin),vmax)
    return encode(x_normalizer,x_perm1)[:,:,1]
end

x = zeros(Float32, nx, ny)
x_init = decode(x_normalizer,reshape(x,nx,ny,1))[:,:,1]

ls = BackTracking(c_1=1f-4,iterations=10,maxstep=Inf32,order=3,ρ_hi=5f-1,ρ_lo=1f-1)
Grad_Loss = zeros(Float32, grad_iterations+1)

T = Float32
vmin = 10f0
vmax = 1000f0

Grad_Loss[1] = f(x)
println("Initial function value: ", Grad_Loss[1])

figure();

init_α = 1f0
for j=1:grad_iterations

    gvec = similar(x)::AbstractArray{T}
    fval = fg!(gvec, x)::T
    p = -gvec/norm(gvec, Inf)

    # linesearch
    function ϕ(α)::T
        try
            fval = f(prj(x .+ α.*p, vmin, vmax))
        catch e
            @assert typeof(e) == DomainError
            fval = T(Inf)
        end
        @show α, fval
        return fval
    end

    α, fval = ls(ϕ, init_α, fval, dot(gvec, p))

    println("inversion iteration no: ",j,"; function value: ",fval)
    Grad_Loss[j+1] = fval

    global x_inv = decode(x_normalizer,reshape(x,nx,ny,1))[:,:,1]
    imshow(x_inv,vmin=20,vmax=120);title("inversion by NN, $j iter");

    # Update model and bound projection
    global x = prj(x .+ α.*p, vmin, vmax)::AbstractArray{T}
    global init_α = α
end

temp2 = decode(y_normalizer,NN(perm_to_tensor(fix_input,nt,grid,dt)))
@assert isapprox(temp1, temp2) # test if network is in test mode (i.e. doesnt' change)

x_true = decode(x_normalizer,x_test_1[:,:,1:1,1,1])[:,:,1]

figure(figsize=(20,12));
subplot(1,3,1)
imshow(x_init,vmin=20,vmax=120);title("initial permeability");
subplot(1,3,2);
imshow(x_inv,vmin=20,vmax=120);title("inversion by NN, $(grad_iterations) iter");
subplot(1,3,3);
imshow(x_true,vmin=20,vmax=120);title("GT permeability");
savefig("result/nn$(grad_iterations)co2.png",bbox_inches="tight",dpi=300)
figure();
plot(Grad_Loss)
