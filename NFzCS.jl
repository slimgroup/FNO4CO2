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

using JOLI, JOLI4Flux
using Printf
using Distributions
using InvertibleNetworks

Random.seed!(1234)

include("utils.jl");
include("fno3dstruct.jl");
include("inversion_utils.jl");
include("pSGLD.jl")
include("pSGD.jl")
include("InvertUtils.jl")

################ Forward -- generate data
JLD2.@load "result/true.jld" eps_ mean_ std_ x_true
n = size(x_true)
nn = prod(n)
subsamp = 0.25f0
A = joRestriction(nn, sort(randperm(nn)[1:Int(round(subsamp*nn))]); DDT=Float32, RDT=Float32)*joRomberg(n[1],n[2]; DDT=Float32, RDT=Float32)
xstar = vec(encode(x_true))
ystar = A * xstar
noise_ = randn(Float32, size(ystar))
snr = 10f0
noise_ = noise_/norm(noise_) *  norm(ystar) * 10f0^(-snr/20f0)
σ = Float32.(norm(noise_)/sqrt(length(noise_)))
yobs = ystar + noise_

grad_iterations = 500
Grad_Loss = zeros(Float32, grad_iterations)

## NF
n_hidden   = 64
L = 4
K = 6
low       = 0.5f0
max_recursion = 1
# Create network
G = NetworkMultiScaleHINT(1, n_hidden, L, K; logdet=false,
                               split_scales=true, max_recursion=max_recursion, p2=0, k2=1, activation=SigmoidLayer(low=low,high=1.0f0))

JLD2.@load "data/TrainedNet/nf.jld2"
P_curr = get_params(G);
for j=1:length(P_curr)
    P_curr[j].data = parameter[j].data;
end
G.logdet = false

G1 = InvertNetRev(G)

λ = 1f0

x = reshape(encode(20f0 * ones(Float32, n)), n[1], n[2], 1, 1)
x_init = decode(x[:,:,1,1])
#x = zeros(Float32, n[1], n[2], 1, 1)
z = vec(G(x))

θ = Flux.params(z)

opt = pSGD(η=5e-2, ρ = 0.99)

figure();

for j=1:grad_iterations

    println("Iteration ", j)
    @time grads = gradient(θ) do
        global x = G1(z)
        misfit = 0.5f0/σ^2f0 * norm(A*vec(x)-yobs)^2f0
        prior = 0.5f0 * λ^2f0 * norm(z)^2f0
        global loss = misfit + prior
        @show misfit, prior
        return loss
    end
    for p in θ
        Flux.Optimise.update!(opt, p, grads[p])
    end
    Grad_Loss[j] = loss
    imshow(decode(x[:,:,1,1]),vmin=20,vmax=120)
end

x_inv = decode(x[:,:,1,1])
figure(figsize=(20,12));
subplot(1,3,1)
imshow(x_init,vmin=20,vmax=120);title("initial permeability");
subplot(1,3,2);
imshow(x_inv,vmin=20,vmax=120);title("inversion by NN w/ NF prior, $grad_iterations iter");
subplot(1,3,3);
imshow(x_true,vmin=20,vmax=120);title("GT permeability");
savefig("result/NFprioronz.png", bbox_inches="tight", dpi=300)

figure();
plot(Grad_Loss)
