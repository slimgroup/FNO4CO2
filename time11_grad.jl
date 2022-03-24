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
using Images
using LineSearches
using SlimPlotting

CUDA.culiteral_pow(::typeof(^), a::Complex{Float32}, b::Val{2}) = real(conj(a)*a)
CUDA.sqrt(a::Complex) = cu(sqrt(a))
Base.broadcasted(::typeof(sqrt), a::Base.Broadcast.Broadcasted) = Base.broadcast(sqrt, Base.materialize(a))

Random.seed!(3)

include("utils.jl");
include("fno3dstruct.jl");
include("inversion_utils.jl")

ntrain = 1000
ntest = 100

BSON.@load "data/TrainedNet/2phasenet_200.bson" NN w batch_size Loss modes width learning_rate epochs gamma step_size;

n = (64,64)
 # dx, dy in m
d = (1f0/64, 1f0/64) # in the training phase

nt = 51
#dt = 20f0    # dt in day
dt = 1f0/nt

perm = matread("data/data/perm.mat")["perm"];
conc = matread("data/data/conc.mat")["conc"];

s = 4

x_train_ = convert(Array{Float32},perm[1:s:end,1:s:end,1:ntrain]);
x_test_ = convert(Array{Float32},perm[1:s:end,1:s:end,end-ntest+1:end]);

nv = 11
survey_indices = Int.(round.(range(2, stop=46, length=nv)))
tsample = (survey_indices .- 1) .* dt

y_train_ = convert(Array{Float32},conc[survey_indices,1:s:end,1:s:end,1:ntrain]);
y_test_ = convert(Array{Float32},conc[survey_indices,1:s:end,1:s:end,end-ntest+1:end]);

y_train_ = permutedims(y_train_,[2,3,1,4]);
y_test = permutedims(y_test_,[2,3,1,4]);

x_normalizer = UnitGaussianNormalizer(x_train_);
x_train_ = encode(x_normalizer,x_train_);
x_test_ = encode(x_normalizer,x_test_);

y_normalizer = UnitGaussianNormalizer(y_train_);
y_train = encode(y_normalizer,y_train_);

x = reshape(collect(range(d[1],stop=n[1]*d[1],length=n[1])), :, 1)
z = reshape(collect(range(d[2],stop=n[2]*d[2],length=n[2])), 1, :)

grid = zeros(Float32,n[1],n[2],2)
grid[:,:,1] = repeat(x',n[2])'
grid[:,:,2] = repeat(z,n[1])

x_train = zeros(Float32,n[1],n[2],nv,4,ntrain)
x_test = zeros(Float32,n[1],n[2],nv,4,ntest)

for i = 1:nv
    x_train[:,:,i,1,:] = deepcopy(x_train_)
    x_test[:,:,i,1,:] = deepcopy(x_test_)
    for j = 1:ntrain
        x_train[:,:,i,2,j] = grid[:,:,1]
        x_train[:,:,i,3,j] = grid[:,:,2]
        x_train[:,:,i,4,j] .= (survey_indices[i]-1)*dt
    end

    for k = 1:ntest
        x_test[:,:,i,2,k] = grid[:,:,1]
        x_test[:,:,i,3,k] = grid[:,:,2]
        x_test[:,:,i,4,k] .= (survey_indices[i]-1)*dt
    end
end

# value, x, y, t
Flux.testmode!(NN, true);
Flux.testmode!(NN.conv1.bn0);
Flux.testmode!(NN.conv1.bn1);
Flux.testmode!(NN.conv1.bn2);
Flux.testmode!(NN.conv1.bn3);

x_test_1 = x_test[:,:,:,:,1:1]
x_test_2 = x_test[:,:,:,:,2:2]
x_test_3 = x_test[:,:,:,:,3:3]

y_test_1 = y_test[:,:,:,1]
y_test_2 = y_test[:,:,:,2]
y_test_3 = y_test[:,:,:,3]

nx, ny = n
dx, dy = d

x_test_1 = x_test[:,:,:,:,1:1]
y_test_1 = y_test[:,:,:,1:1]

grad_iterations = 50
std_ = x_normalizer.std_[:,:,1]
eps_ = x_normalizer.eps_
mean_ = x_normalizer.mean_[:,:,1]

function f(x_inv)
    println("evaluate f")
    @time begin
        out = decode(y_normalizer,NN(perm_to_tensor(x_inv,tsample,grid)))
        misfit = 0.5f0 * (nt-1)/(nv-1) * norm(out-y_test_1)^2f0
        prior = 0.5f0 * λ * norm(x_inv)^2f0
        loss = misfit + prior
        @show loss, misfit, prior
    end
    return loss
end

function g!(gvec, x_inv)
    println("evaluate g")
    p = params(x_inv)
    @time grads = gradient(p) do
        out = decode(y_normalizer,NN(perm_to_tensor(x_inv,tsample,grid)))
        misfit = 0.5f0 * (nt-1)/(nv-1) * norm(out-y_test_1)^2f0
        prior = 0.5f0 * λ * norm(x_inv)^2f0
        global loss = misfit + prior
        @show loss, misfit, prior
        return loss
    end
    copyto!(gvec, grads.grads[x_inv])
end

function fg!(gvec, x_inv)
    println("evaluate f and g")
    p = params(x_inv)
    @time grads = gradient(p) do
        out = decode(y_normalizer,NN(perm_to_tensor(x_inv,tsample,grid)))
        misfit = 0.5f0 * (nt-1)/(nv-1) * norm(out-y_test_1)^2f0
        prior = 0.5f0 * λ * norm(x_inv)^2f0
        global loss = misfit + prior
        @show loss, misfit, prior
        return loss
    end
    copyto!(gvec, grads.grads[x_inv])
    return loss
end

λ = 0f0 # 2 norm regularization
#x = encode(x_normalizer,20f0*ones(Float32,nx,ny))[:,:,1]
x = zeros(Float32, nx, ny)
x_init = decode(x_normalizer,reshape(x,nx,ny,1))[:,:,1]

ls = BackTracking(c_1=1f-4,iterations=10,maxstep=Inf32,order=3,ρ_hi=5f-1,ρ_lo=1f-1)
Grad_Loss = zeros(Float32, grad_iterations)

T = Float32

function prj(x; vmin=10f0, vmax=130f0)
    y = decode(x)
    z = max.(min.(y,vmax),vmin)
    return encode(z)
end

println("Initial function value: ", f(x))
fig, ax = subplots(nrows=1,ncols=1,figsize=(20,12))
x_true = decode(x_normalizer,x_test_1[:,:,1:1,1,1])[:,:,1]
plot_velocity(x_true, d; vmin=10f0, vmax=130f0, ax=ax, new_fig=false, name="ground truth");

fig, ax = subplots(nrows=1,ncols=1,figsize=(20,12))
for j=1:grad_iterations

    gvec = similar(x)::AbstractArray{T}
    fval = fg!(gvec, x)::T
    p = -gvec/norm(gvec, Inf)

    # linesearch
    function ϕ(α)::T
        try
            fval = f(prj(x .+ α.*p))
        catch e
            @assert typeof(e) == DomainError
            fval = T(Inf)
        end
        @show α, fval
        return fval
    end

    α, fval = ls(ϕ, 1f0, fval, dot(gvec, p))

    println("inversion iteration no: ",j,"; function value: ",fval)
    Grad_Loss[j] = fval

    # Update model and bound projection
    global x = prj(x .+ α.*p)::AbstractArray{T}
    #global λ = Float32(λ*sqrt(j)/sqrt(j+1))
    #if j>=20
    #    global λ = 0f0
    #end

    plot_velocity(decode(x), d; vmin=10f0, vmax=130f0, ax=ax, new_fig=false, name="inversion after $j iterations");
    ax.set_title("inversion by NN, $j iter");
end


figure(figsize=(20,12));
subplot(1,3,1)
imshow(x_init,vmin=20,vmax=120);title("initial permeability");
subplot(1,3,2);
imshow(x_inv,vmin=20,vmax=120);title("inversion by NN, $(grad_iterations) iter");
subplot(1,3,3);
imshow(x_true,vmin=20,vmax=120);title("GT permeability");

figure();
plot(Grad_Loss)
