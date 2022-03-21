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

CUDA.culiteral_pow(::typeof(^), a::Complex{Float32}, b::Val{2}) = real(conj(a)*a)
CUDA.sqrt(a::Complex) = cu(sqrt(a))
Base.broadcasted(::typeof(sqrt), a::Base.Broadcast.Broadcasted) = Base.broadcast(sqrt, Base.materialize(a))

include("utils.jl");
include("fno3dstruct.jl");

Random.seed!(3)

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
y_test_1 = deepcopy(conc[:,1:subsample:end,1:subsample:end,1001]);

################ Forward -- generate data

nv = 11

survey_indices = Int.(round.(range(1, stop=nt, length=nv)))

sw = y_test_1[survey_indices,:,:,1]

n = (size(sw,3), size(sw,2))

vp = 3500 * ones(Float32,n)
vs = vp ./ sqrt(3f0)
phi = 0.25f0 * ones(Float32,n)

rho = 2200 * ones(Float32,n)

vp_stack = [(Patchy(sw[i,:,:]',vp,vs,rho,phi))[1] for i = 1:nv]

##### Wave equation

d = (15f0, 15f0)
o = (0f0, 0f0)

extentx = (n[1]-1)*d[1]
extentz = (n[2]-1)*d[2]

nsrc = 15
nrec = n[2]

model = [Model(n, d, o, (1000f0 ./ vp_stack[i]).^2f0; nb = 80) for i = 1:nv]

timeS = timeR = 750f0
dtS = dtR = 1f0
ntS = Int(floor(timeS/dtS))+1
ntR = Int(floor(timeR/dtR))+1

xsrc = convertToCell(range(5*d[1],stop=5*d[1],length=nsrc))
ysrc = convertToCell(range(0f0,stop=0f0,length=nsrc))
zsrc = convertToCell(range(d[2],stop=(n[2]-1)*d[2],length=nsrc))

xrec = range((n[1]-5)*d[1],stop=(n[1]-5)*d[1], length=nrec)
yrec = 0f0
zrec = range(d[2],stop=(n[2]-1)*d[2],length=nrec)

srcGeometry = Geometry(xsrc, ysrc, zsrc; dt=dtS, t=timeS)
recGeometry = Geometry(xrec, yrec, zrec; dt=dtR, t=timeR, nsrc=nsrc)

f0 = 0.02f0     # kHz
wavelet = ricker_wavelet(timeS, dtS, f0)
q = judiVector(srcGeometry, wavelet)

ntComp = get_computational_nt(srcGeometry, recGeometry, model[end])
info = Info(prod(n), nsrc, ntComp)

opt = Options(return_array=true)
Pr = judiProjection(info, recGeometry)
Ps = judiProjection(info, srcGeometry)

F = [Pr*judiModeling(info, model[i]; options=opt)*Ps' for i = 1:nv]

d_obs = [F[i]*q for i = 1:nv]

JLD2.@save "data/data/time_lapse_data.jld2" d_obs

G = Forward(F[1],q)

x_perm = 20*ones(Float32,n[1],n[2],1)

grad_iterations = 200

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

function f(x_inv)
    println("evaluate f")
    @time begin
        sw = decode(y_normalizer,NN(perm_to_tensor(x_inv,nt,grid,dt)))
        vp_stack = [(Patchy(sw[:,:,survey_indices[i],1]',vp,vs,rho,phi))[1] for i = 1:nv]
        m_stack = [(1000f0 ./ vp_stack[i]).^2f0 for i = 1:nv]
        d_predict = [G(m_stack[i]) for i = 1:nv]
        loss = 0.5f0 * norm(d_predict-d_obs)^2f0/nv/nsrc
    end
    return loss
end

function g!(gvec, x_inv)
    println("evaluate g")
    p = params(x_inv)
    @time grads = gradient(p) do
        sw = decode(y_normalizer,NN(perm_to_tensor(x_inv,nt,grid,dt)))
        vp_stack = [(Patchy(sw[:,:,survey_indices[i],1]',vp,vs,rho,phi))[1] for i = 1:nv]
        m_stack = [(1000f0 ./ vp_stack[i]).^2f0 for i = 1:nv]
        d_predict = [G(m_stack[i]) for i = 1:nv]
        loss = 0.5f0 * norm(d_predict-d_obs)^2f0/nv/nsrc
        return loss
    end
    copyto!(gvec, grads.grads[x_inv])
end

function fg!(gvec, x_inv)
    println("evaluate f and g")
    p = params(x_inv)
    @time grads = gradient(p) do
        sw = decode(y_normalizer,NN(perm_to_tensor(x_inv,nt,grid,dt)))
        vp_stack = [(Patchy(sw[:,:,survey_indices[i],1]',vp,vs,rho,phi))[1] for i = 1:nv]
        m_stack = [(1000f0 ./ vp_stack[i]).^2f0 for i = 1:nv]
        d_predict = [G(m_stack[i]) for i = 1:nv]
        global loss = 0.5f0 * norm(d_predict-d_obs)^2f0/nv/nsrc
        return loss
    end
    copyto!(gvec, grads.grads[x_inv])
    return loss
end

x = zeros(Float32, nx, ny)

ls = BackTracking(c_1=1f-4,iterations=25,maxstep=Inf32,order=3,ρ_hi=5f-1,ρ_lo=1f-1)
Grad_Loss = zeros(Float32, grad_iterations+1)

T = Float32

println("Initial function value: ", f(x))

for j=1:grad_iterations

    gvec = similar(x)::AbstractArray{T}
    fval = fg!(gvec, x)::T
    p = -gvec/norm(gvec, Inf)

    # linesearch
    function ϕ(α)::T
        try
            fval = f(x .+ α.*p)
        catch e
            @assert typeof(e) == DomainError
            fval = T(Inf)
        end
        @show α, fval
        return fval
    end

    α, fval = ls(ϕ, 1f0, fval, dot(gvec, p))

    println("Coupled inversion iteration no: ",j,"; function value: ",fval)
    Grad_Loss[j] = fval
    curr_loss = Grad_Loss[1:j]
    # Update model and bound projection
    @. x = x + α*p::AbstractArray{T}
    JLD2.@save "result/SEGnocheckpointiter$(j)iter.jld2" x curr_loss α p
end

JLD2.@save "result/SEGnocheckpoint.jld2" Grad_Loss