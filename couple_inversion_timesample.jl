# author: Ziyi Yin
# This code is an implementation of fourier neural operators from Zongyi Li's repository

using DrWatson
@quickactivate "FNO"
import Pkg; Pkg.instantiate()

ENV["MPLBACKEND"]="qt5agg"
using PyPlot
using BSON
using Flux, Random, FFTW, Zygote, NNlib
using MAT, Statistics, LinearAlgebra
using CUDA
using ProgressMeter, JLD2
using Images
using LineSearches
using JUDI, JUDI4Flux
using SlimPlotting

CUDA.culiteral_pow(::typeof(^), a::Complex{Float32}, b::Val{2}) = real(conj(a)*a)
CUDA.sqrt(a::Complex) = cu(sqrt(a))
Base.broadcasted(::typeof(sqrt), a::Base.Broadcast.Broadcasted) = Base.broadcast(sqrt, Base.materialize(a))

Random.seed!(3)

include("utils.jl");
include("fno3dstruct.jl");
include("inversion_utils.jl");
include("pSGLD.jl")
include("pSGD.jl")
include("InvertUtils.jl")
include("illum.jl")

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

s = 4

x_train_ = convert(Array{Float32},perm[1:s:end,1:s:end,1:ntrain]);
x_test_ = convert(Array{Float32},perm[1:s:end,1:s:end,end-ntest+1:end]);

nv = 11
survey_indices = Int.(round.(range(1, stop=36, length=nv)))
tsample = (survey_indices .- 1) .* dt

y_train_ = convert(Array{Float32},conc[survey_indices,1:s:end,1:s:end,1:ntrain]);
y_test_ = convert(Array{Float32},conc[survey_indices,1:s:end,1:s:end,end-ntest+1:end]);

y_train_ = permutedims(y_train_,[2,3,1,4]);
y_test = permutedims(y_test_,[2,3,1,4]);

x_normalizer = UnitGaussianNormalizer(x_train_);
x_train_ = encode(x_normalizer,x_train_);
x_test_ = encode(x_normalizer,x_test_);

y_normalizer = UnitGaussianNormalizer(y_train_);
y_train = encode(y_normalizer,y_train_);

x = reshape(collect(range(d[1],stop=n[1]*d[1],length=n[1])), :, 1)
z = reshape(collect(range(d[2],stop=n[2]*d[2],length=n[2])), 1, :)

grid = zeros(Float32,n[1],n[2],2)
grid[:,:,1] = repeat(x',n[2])'
grid[:,:,2] = repeat(z,n[1])

x_train = zeros(Float32,n[1],n[2],nv,4,ntrain)
x_test = zeros(Float32,n[1],n[2],nv,4,ntest)

for i = 1:nv
    x_train[:,:,i,1,:] = deepcopy(x_train_)
    x_test[:,:,i,1,:] = deepcopy(x_test_)
    for j = 1:ntrain
        x_train[:,:,i,2,j] = grid[:,:,1]
        x_train[:,:,i,3,j] = grid[:,:,2]
        x_train[:,:,i,4,j] .= (survey_indices[i]-1)*dt
    end

    for k = 1:ntest
        x_test[:,:,i,2,k] = grid[:,:,1]
        x_test[:,:,i,3,k] = grid[:,:,2]
        x_test[:,:,i,4,k] .= (survey_indices[i]-1)*dt
    end
end

# value, x, y, t
Flux.testmode!(NN, true);
Flux.testmode!(NN.conv1.bn0);
Flux.testmode!(NN.conv1.bn1);
Flux.testmode!(NN.conv1.bn2);
Flux.testmode!(NN.conv1.bn3);

x_test_1 = x_test[:,:,:,:,1:1]
x_test_2 = x_test[:,:,:,:,2:2]
x_test_3 = x_test[:,:,:,:,3:3]

y_test_1 = y_test[:,:,:,1]
y_test_2 = y_test[:,:,:,2]
y_test_3 = y_test[:,:,:,3]

nx, ny = n
dx, dy = d

x_test_1 = x_test[:,:,:,:,1:1]
y_test_1 = y_test[:,:,:,1:1]

grad_iterations = 400
std_ = x_normalizer.std_[:,:,1]
eps_ = x_normalizer.eps_
mean_ = x_normalizer.mean_[:,:,1]

vp = 3500 * ones(Float32,n)
vs = vp ./ sqrt(3f0)
phi = 0.25f0 * ones(Float32,n)

rho = 2200 * ones(Float32,n)

vp_stack = [(Patchy(y_test_1[:,:,i,1]',vp,vs,rho,phi))[1] for i = 1:nv]

##### Wave equation

d = (15f0, 15f0)
o = (0f0, 0f0)

extentx = (n[1]-1)*d[1]
extentz = (n[2]-1)*d[2]

nsrc = 32
nrec = Int(round((n[2]-1)*d[2]))

model = [Model(n, d, o, (1f3 ./ vp_stack[i]).^2f0; nb = 80) for i = 1:nv]

timeS = timeR = 750f0
dtS = dtR = 1f0
ntS = Int(floor(timeS/dtS))+1
ntR = Int(floor(timeR/dtR))+1

xsrc = convertToCell(range(1*d[1],stop=1*d[1],length=nsrc))
ysrc = convertToCell(range(0f0,stop=0f0,length=nsrc))
zsrc = convertToCell(range(d[2],stop=(n[2]-1)*d[2],length=nsrc))

xrec = range((n[1]-1)*d[1],stop=(n[1]-1)*d[1], length=nrec)
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

try
    JLD2.@load "data/data/time_lapse_data_$(nv)nv_$(nsrc)nsrc.jld2" d_obs
    global d_obs = d_obs
    println("found data, loading")
catch e
    println("generating")
    global d_obs = [F[i]*q for i = 1:nv]
    JLD2.@save "data/data/time_lapse_data_$(nv)nv_$(nsrc)nsrc.jld2" d_obs
end

λ = 1f-1 # 2 norm regularization

#x = encode(x_normalizer,20f0*ones(Float32,nx,ny))[:,:,1]
x = zeros(Float32, nx, ny)
x_init = decode(x_normalizer,reshape(x,nx,ny,1))[:,:,1]

function prj(x; vmin=10f0, vmax=130f0)
    y = decode(x)
    z = max.(min.(y,vmax),vmin)
    return encode(z)
end

fig, ax = subplots(nrows=1,ncols=1,figsize=(20,12))
figloss, axloss = subplots(nrows=1,ncols=1,figsize=(20,12));axloss.set_title("loss");
figmisfit, axmisfit = subplots(nrows=1,ncols=1,figsize=(20,12));axmisfit.set_title("misfit");
figprior, axprior = subplots(nrows=1,ncols=1,figsize=(20,12));axprior.set_title("prior");

hisloss = zeros(Float32, grad_iterations)
hismisfit = zeros(Float32, grad_iterations)
hisprior = zeros(Float32, grad_iterations)
nssample = 4

fval = Inf32
misfit = Inf32
prior = Inf32
opt = Flux.Optimise.ADAMW(1f-1, (0.9f0, 0.999f0), 1f-4)

for j=1:grad_iterations

    println("Iteration ", j)
    rand_ns = [jitter(nsrc, nssample) for i = 1:nv]
    G_stack = [ForwardIllum(F[i][rand_ns[i]],q[rand_ns[i]]) for i = 1:nv]
    d_obs_sample = [sample_src(d_obs[i], nsrc, rand_ns[i]) for i = 1:nv]

    θ = Flux.params(x)
    @time grads = gradient(θ) do
        sw = decode(y_normalizer,NN(perm_to_tensor(x,tsample,grid)))
        vp_stack = [(Patchy(sw[:,:,i,1]',vp,vs,rho,phi))[1] for i = 1:nv]
        d_predict = [G_stack[i]((1000f0 ./ vp_stack[i]).^2f0) for i = 1:nv]
        global misfit = Float32(0.5f0/nssample/nv) * norm(d_predict-d_obs_sample).^2f0
        global prior = 0.5f0 * λ^2f0 * sum(x.^2f0)
        global fval = misfit + prior
        @show misfit, prior
        return fval
    end
    for p in θ
        Flux.Optimise.update!(opt, p, grads[p])
    end
    println("Coupled inversion iteration no: ",j,"; function value: ",fval)
    hisloss[j] = fval
    hismisfit[j] = misfit
    hisprior[j] = prior

    # Update model and bound projection
    global x = prj(x)::Matrix{Float32}
    plot_velocity(decode(x), d; vmin=10f0, vmax=130f0, ax=ax, new_fig=false, name="inversion after $j iterations");
    axloss.plot(hisloss[1:j])
    axmisfit.plot(hismisfit[1:j])
    axprior.plot(hisprior[1:j])
    println("Coupled inversion iteration no: ",j,"; function value: ",fval)

end

figure(figsize=(20,12));
subplot(1,3,1)
imshow(x_init,vmin=20,vmax=120);title("initial permeability");
subplot(1,3,2);
imshow(decode(x),vmin=20,vmax=120);title("inversion by NN, $(grad_iterations) iter");
subplot(1,3,3);
imshow(x_true,vmin=20,vmax=120);title("GT permeability");

figure();
plot(hisloss)

