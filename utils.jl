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

function decode(x::Matrix{Float32})
    x1 = x .* (std_ .+ eps_) .+ mean_
    return x1
end

function encode(x::Matrix{Float32})
    x1 = (x .- mean_) ./ (std_ .+ eps_) 
    return x1
end