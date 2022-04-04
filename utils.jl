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

function loadPairs()

# load training pairs
perm = matread("data/data/perm.mat")["perm"];
conc = matread("data/data/conc.mat")["conc"];

# spatial downsampling
s = 4

x_train_ = convert(Array{Float32},perm[1:s:end,1:s:end,1:ntrain])
x_test_ = convert(Array{Float32},perm[1:s:end,1:s:end,end-ntest+1:end])

y_train_ = convert(Array{Float32},conc[:,1:s:end,1:s:end,1:ntrain])
y_test_ = convert(Array{Float32},conc[:,1:s:end,1:s:end,end-ntest+1:end])

y_train_ = permutedims(y_train_,[2,3,1,4])
y_test = permutedims(y_test_,[2,3,1,4])

x_normalizer = UnitGaussianNormalizer(x_train_)
x_train_ = encode(x_normalizer,x_train_)
x_test_ = encode(x_normalizer,x_test_)

y_normalizer = UnitGaussianNormalizer(y_train_)
y_train = encode(y_normalizer,y_train_)

x = reshape(collect(range(d[1],stop=n[1]*d[1],length=n[1])), :, 1)
z = reshape(collect(range(d[2],stop=n[2]*d[2],length=n[2])), 1, :)

grid = zeros(Float32,n[1],n[2],2)
grid[:,:,1] = repeat(x',n[2])'
grid[:,:,2] = repeat(z,n[1])

x_train = zeros(Float32,n[1],n[2],nt,4,ntrain)
x_test = zeros(Float32,n[1],n[2],nt,4,ntest)

for i = 1:nt
    x_train[:,:,i,1,:] = deepcopy(x_train_)
    x_test[:,:,i,1,:] = deepcopy(x_test_)
    for j = 1:ntrain
        x_train[:,:,i,2,j] = grid[:,:,1]
        x_train[:,:,i,3,j] = grid[:,:,2]
        x_train[:,:,i,4,j] .= (i-1)*dt
    end

    for k = 1:ntest
        x_test[:,:,i,2,k] = grid[:,:,1]
        x_test[:,:,i,3,k] = grid[:,:,2]
        x_test[:,:,i,4,k] .= (i-1)*dt
    end
end
return x_train, x_test, y_train, y_test, x_normalizer, y_normalizer, grid
end

function setMeanStd(x_normalizer)
    std_ = x_normalizer.std_[:,:,1]
    eps_ = x_normalizer.eps_
    mean_ = x_normalizer.mean_[:,:,1]
    return std_, eps_, mean_
end