using DrWatson
@quickactivate "FNO"
import Pkg; Pkg.instantiate()

using PyPlot
using Flux, Random, FFTW, Zygote, NNlib
using MAT, Statistics, LinearAlgebra
using TestImages

include("pSGLD.jl")
include("cnn.jl")
Random.seed!(3)

layers = DeepPriorNet()
G(z) = DeepPriorNet(z, layers)
Flux.trainmode!(layers, true)

n = (64, 64)
img = Float32.(TestImages.shepp_logan(n[1]))
noise_ = randn(Float32, size(img))
snr = 20f0
noise_ = noise_/norm(noise_) *  norm(img) * 10f0^(-snr/20f0)
σ = Float32.(norm(noise_)/sqrt(length(noise_)))
imgobs = img + noise_

η = 1f-3
opt = pSGLD(η)

z = randn(Float32, n[1], n[2], 3, 1)
Flux.testmode!(layers, true)
x = G(z)[:,:,1,1]
Flux.trainmode!(layers, true)

grad_iterations = 1000
Grad_Loss = zeros(Float32, grad_iterations)
w = Flux.params(layers)

samples_1k = zeros(Float32, n[1], n[2], 1000)
loss_1k = zeros(Float32, n[1], n[2], 1000)

λ = √0f0

figure();
for j=1:grad_iterations

    println("Iteration ", j)
    @time grads = gradient(w) do
        global x_inv = G(z)[:,:,1,1]
        misfit = 0.5f0/σ^2f0 * norm(x_inv-imgobs)^2f0
        prior = 0.5f0 * λ^2f0 * norm(w)^2f0
        global loss = misfit + prior
        @show misfit, prior
        return loss
    end
    Grad_Loss[j] = loss
    for p in w
        Flux.Optimise.update!(opt, p, grads[p])
    end
    imshow(x_inv);title("deep prior denoising after $j iterations")
end

#JLD2.@save "result/pSGLD.jld2" Grad_Loss