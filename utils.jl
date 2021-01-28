export UnitGaussianNormalizer

struct UnitGaussianNormalizer
    mean_
    std_
    eps_
end

function UnitGaussianNormalizer(x;eps_=1f-5)
    mean_ = mean(x,dims=3)
    std_ = std(x,dims=3)
    return UnitGaussianNormalizer(mean_,std_,eps_)
end

function encode(normalizer::UnitGaussianNormalizer,x)
    x = (x.-normalizer.mean_)./(normalizer.std_.+normalizer.eps_)
end

function decode(normalizer::UnitGaussianNormalizer,x;sample_idx=nothing)
    if isnothing(sample_idx)
        std_ = normalizer.std_ .+ normalizer.eps_
        mean_ = normalizer.mean_
    else
        if length(size(normalizer.mean_)) == length(size(sample_idx[0]))
            std_ = normalizer.std_[sample_idx] .+ normalizer.eps_
            mean_ = normalizer.mean_[sample_idx]
        end
        if length(size(normalizer.mean_)) > length(size(sample_idx[0]))
            std_ = normalizer.std_[:,sample_idx] .+ normalizer.eps_
            mean_ = normalizer.mean_[:,sample_idx]
        end
    end
    x = x .* std_ .+ mean_
    return x
end
