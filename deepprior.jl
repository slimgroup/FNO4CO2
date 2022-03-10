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

include("utils.jl");
include("fno3dstruct.jl");
include("inversion_utils.jl");
include("cnn.jl")
include("pSGLD.jl")
include("pSGD.jl")

Random.seed!(3)

layers = DeepPriorNet()
G(z) = DeepPriorNet(z, layers)
Flux.trainmode!(layers, true)

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

subsample = 4

x_train_ = convert(Array{Float32},perm[1:subsample:end,1:subsample:end,1:ntrain]);
x_test_ = convert(Array{Float32},perm[1:subsample:end,1:subsample:end,end-ntest+1:end]);

y_train_ = convert(Array{Float32},conc[:,1:subsample:end,1:subsample:end,1:ntrain]);
y_test_ = convert(Array{Float32},conc[:,1:subsample:end,1:subsample:end,end-ntest+1:end]);

y_train_ = permutedims(y_train_,[2,3,1,4]);
y_test = permutedims(y_test_,[2,3,1,4]);

x_normalizer = UnitGaussianNormalizer(x_train_);
x_train_ = encode(x_normalizer,x_train_);
x_test_ = encode(x_normalizer,x_test_);

y_normalizer = UnitGaussianNormalizer(y_train_);
y_train = encode(y_normalizer,y_train_);

x = reshape(collect(range(d[1],stop=n[1]*d[1],length=n[1])), :, 1);
z = reshape(collect(range(d[2],stop=n[2]*d[2],length=n[2])), 1, :);

grid = zeros(Float32,n[1],n[2],2);
grid[:,:,1] = repeat(x',n[2])';
grid[:,:,2] = repeat(z,n[1]);

x_train = zeros(Float32,n[1],n[2],nt,4,ntrain);
x_test = zeros(Float32,n[1],n[2],nt,4,ntest);

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
x_test_1 = deepcopy(perm[1:subsample:end,1:subsample:end,1001]);

################ Forward -- generate data

sw_true = y_test[:,:,:,1]
nwells = 15
obsloc = Int.(round.(range(1, stop=n[2], length=nwells)))
swobs_true = sw_true[:,obsloc,:]
noise_ = randn(Float32, size(swobs_true))
snr = 20f0
noise_ = noise_/norm(noise_) *  norm(swobs_true) * 10f0^(-snr/20f0)
σ = Float32.(norm(noise_)/sqrt(length(noise_)))
swobs = swobs_true + noise_

λ = √1f0

opt = pSGD()

z = randn(Float32, n[1], n[2], 3, 1)
Flux.testmode!(layers, true)
x = G(z)[:,:,1,1]
Flux.trainmode!(layers, true)

w = Flux.params(layers)

samples_1k = zeros(Float32, n[1], n[2], 1000)
loss_1k = zeros(Float32, n[1], n[2], 1000)

### pre-train to 0
for j=1:100

    println("Iteration ", j)
    @time grads = gradient(w) do
        global x_inv = G(z)[:,:,1,1]
        global loss = 0.5f0 * norm(x_inv)^2f0
        @show loss
        return loss
    end
    for p in w
        Flux.Optimise.update!(opt, p, grads[p])
    end
end
x_init = decode(x_normalizer, reshape(x, nx, ny, 1))[:,:,1]

### SGLD
grad_iterations = 10000
Grad_Loss = zeros(Float32, grad_iterations)
opt = pSGLD()

figure()
for j=1:grad_iterations

    println("Iteration ", j)
    @time grads = gradient(w) do
        global x_inv = G(z)[:,:,1,1]
        sw = decode(y_normalizer,NN(perm_to_tensor(x_inv,nt,grid,dt)))
        misfit = 0.5f0/σ^2f0 * norm(sw[:,obsloc,:,1]-swobs)^2f0
        prior = 0.5f0 * λ^2f0 * norm(w)^2f0
        global loss = misfit + prior
        @show misfit, prior
        return loss
    end
    Grad_Loss[j] = loss
    for p in w
        Flux.Optimise.update!(opt, p, grads[p])
    end
    if j%1000==0
        loss_1k[1000] = loss
        samples_1k[:,:,1000] = decode(x_normalizer,reshape(G(z)[:,:,1,1],nx,ny,1))[:,:,1]
        JLD2.@save "result/SGLD$(j-999)to$(j)samples.jld2" loss_1k samples_1k
    else
        loss_1k[j%1000] = loss
        samples_1k[:,:,j%1000] = decode(x_normalizer,reshape(G(z)[:,:,1,1],nx,ny,1))[:,:,1]
    end
    imshow(decode(x_normalizer,G(z)[:,:,1,1])[:,:,1], vmin=20, vmax=120)
end

JLD2.@save "result/pSGLD.jld2" Grad_Loss