export UnitGaussianNormalizer

mutable struct UnitGaussianNormalizer
    mean_ 
    std_
    eps_
end

function UnitGaussianNormalizer(x;eps_=1f-5)
    mean_ = mean(x,dims=length(size(x)))
    std_ = std(x,dims=length(size(x)))
    return UnitGaussianNormalizer(mean_,std_,eps_)
end

function encode(normalizer::UnitGaussianNormalizer,x)
    x1 = (x.-normalizer.mean_)./(normalizer.std_.+normalizer.eps_)
    return x1
end

function decode(normalizer::UnitGaussianNormalizer,x;sample_idx=nothing)
    std_ = normalizer.std_ .+ normalizer.eps_
    mean_ = normalizer.mean_
    x1 = x .* std_ .+ mean_
    return x1
end

function decode(x::AbstractMatrix{Float32})
    x1 = x .* (std_ .+ eps_) .+ mean_
    return x1
end

function encode(x::AbstractMatrix{Float32})
    x1 = (x .- mean_) ./ (std_ .+ eps_) 
    return x1
end