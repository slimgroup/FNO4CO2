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

using JUDI, JUDI4Flux
using JOLI
using Printf
using Distributions
using InvertibleNetworks

CUDA.culiteral_pow(::typeof(^), a::Complex{Float32}, b::Val{2}) = real(conj(a)*a)
CUDA.sqrt(a::Complex) = cu(sqrt(a))
Base.broadcasted(::typeof(sqrt), a::Base.Broadcast.Broadcasted) = Base.broadcast(sqrt, Base.materialize(a))

include("utils.jl");
include("fno3dstruct.jl");
include("inversion_utils.jl");
include("pSGLD.jl")
include("pSGD.jl")
include("InvertUtils.jl")

Random.seed!(3)

ntrain = 1000
ntest = 100

BSON.@load "data/TrainedNet/2phasenet_200.bson" NN w batch_size Loss modes width learning_rate epochs gamma step_size;

n_hidden   = 64
L = 4
K = 6
low       = 0.5f0
max_recursion = 1
# Create network
G = NetworkMultiScaleHINT(1, n_hidden, L, K;
                               split_scales=true, max_recursion=max_recursion, p2=0, k2=1, activation=SigmoidLayer(low=low,high=1.0f0))

JLD2.@load "data/TrainedNet/nf.jld2"
P_curr = get_params(G);
for j=1:length(P_curr)
    P_curr[j].data = parameter[j].data;
end

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
y_test_1 = deepcopy(conc[:,1:subsample:end,1:subsample:end,1001]);

################ Forward -- generate data

sw_true = y_test[:,:,:,1]
nwells = 64
obsloc = Int.(round.(range(1, stop=nt, length=nwells)))
swobs_true = sw_true[:,obsloc,:]
noise_ = randn(Float32, size(swobs_true))
snr = 20f0
noise_ = noise_/norm(noise_) *  norm(swobs_true) * 10f0^(-snr/20f0)
σ = Float32.(norm(noise_)/sqrt(length(noise_)))
swobs = swobs_true + noise_

grad_iterations = 50
Grad_Loss = zeros(Float32, grad_iterations)

T = Float32
vmin = 10f0
vmax = 130f0

figure();

λ = 1f0

x = zeros(Float32, nx, ny, 1, 1)
x_init = decode(x_normalizer, reshape(x, nx, ny, 1))[:,:,1]

G1 = InvertNetRev(G)

z = vec(G(x)[1])
θ = Flux.params(z)

opt = pSGD(η=1e-2, ρ = 0.99)

for j=1:grad_iterations

    println("Iteration ", j)
    @time grads = gradient(θ) do
        global x = G1(z)
        sw = decode(y_normalizer,NN(perm_to_tensor(x[:,:,1,1],nt,grid,dt)))
        misfit = 0.5f0/σ^2f0 * norm(sw[:,obsloc,:,1]-swobs)^2f0
        prior = 0.5f0 * λ^2f0 * norm(z)^2f0
        global loss = misfit + prior
        @show misfit, prior
        return loss
    end
    for p in θ
        Flux.Optimise.update!(opt, p, grads[p])
    end
    Grad_Loss[j] = loss
    imshow(decode(x_normalizer,x)[:,:,1],vmin=20,vmax=120)
end

x_inv = decode(x_normalizer,x)[:,:,1]
figure();
subplot(1,3,1)
imshow(x_init,vmin=20,vmax=120);title("initial permeability");
subplot(1,3,2);
imshow(x_inv,vmin=20,vmax=120);title("inversion by NN w/ NF prior, $grad_iterations iter");
subplot(1,3,3);
imshow(x_test_1,vmin=20,vmax=120);title("GT permeability");
savefig("result/NFpriorfromsat.png", bbox_inches="tight", dpi=300)

figure();
plot(Grad_Loss)
