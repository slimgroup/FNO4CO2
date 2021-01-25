# author: Ziyi Yin
# This code is an implementation of fourier neural operators from Zongyi Li's repository

using PyPlot
using Flux, Random

Random.seed!(3)

mutable struct FourierNeuralOperatorNet2D
    modes::Int
    width::Int
end

@Flux.functor FourierNeuralOperatorNet2D