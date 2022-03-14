# Author: Ali Siahkoohi, alisk@gatech.edu
# Date: September 2020
# Copyright: Georgia Institute of Technology, 2020

export update!, pSGLD

import Flux.Optimise: update!
using Flux: Params

const ϵ = 1e-8

    """
    pSGLD(η = 0.001, ρ = 0.99)
    Optimizer using the [pSGLD](ttps://arxiv.org/pdf/1512.07666.pdf) algorithm.
    Built on the Flux RMSprop implementation at:
    https://github.com/FluxML/Flux.jl/blob/master/src/optimise/optimisers.jl#L137)
 *Input*:
 - Learning rate (`η`): Amount by which gradients are discounted before updating
                        the weights.
 - `Momentum (`ρ`): Controls the acceleration of gradient descent in the
                    prominent direction, in effect dampening oscillations.
    """

mutable struct pSGLD
  eta::Float64
  rho::Float64
  acc::IdDict
end

pSGLD(;η = 0.001, ρ = 0.99) = pSGLD(η, ρ, IdDict())

function apply!(o::pSGLD, x, Δ)
    η, ρ = o.eta, o.rho
    acc = get!(o.acc, x, zero(x))::typeof(x)
    @. acc = ρ * acc + (1 - ρ) * Δ^2
    @. Δ *= η / (√acc + ϵ)
    n = randn(size(Δ))
    if typeof(n) == Array{Float64,0} || typeof(n) == Array{Float32,0}
        n = n[1]
    end
    @. Δ += convert(typeof(Δ), n).*√(2*η / (√acc + ϵ))
    return @. Δ
end

function update!(opt::pSGLD, x, x̄)
    x .-= apply!(opt, x, x̄)
end

function update!(opt::pSGLD, xs::Params, gs)
    for x in xs
        isnothing(gs[x]) && continue
        update!(opt, x, gs[x])
    end
end