# author: Ziyi Yin
# This code is an implementation of fourier neural operators from Zongyi Li's repository

using DrWatson
@quickactivate "FNO"
import Pkg; Pkg.instantiate()

ENV["MPLBACKEND"]="agg"
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
using SetIntersectionProjection

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

s = 4
st = 2

n = (64,64) # dx, dy in m
d = (1f0/64, 1f0/64) # in the training phase

nt = 26
dt = 2f0/51f0

perm = matread("data/data/perm.mat")["perm"];
conc = matread("data/data/conc.mat")["conc"];

x_train_ = convert(Array{Float32},perm[1:s:end,1:s:end,1:ntrain]);
x_test_ = convert(Array{Float32},perm[1:s:end,1:s:end,end-ntest+1:end]);

y_train_ = convert(Array{Float32},conc[1:st:end,1:s:end,1:s:end,1:ntrain]);
y_test_ = convert(Array{Float32},conc[1:st:end,1:s:end,1:s:end,end-ntest+1:end]);

nv = 11
survey_indices = Int.(round.(range(1, stop=nt, length=nv)))

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

# value, x, y, t
Flux.testmode!(NN, true);
Flux.testmode!(NN.conv1.bn0);
Flux.testmode!(NN.conv1.bn1);
Flux.testmode!(NN.conv1.bn2);
Flux.testmode!(NN.conv1.bn3);

sw_true = y_test[:,:,survey_indices,16]

nx, ny = n
dx, dy = d

grad_iterations = 120
std_ = x_normalizer.std_[:,:,1]
eps_ = x_normalizer.eps_
mean_ = x_normalizer.mean_[:,:,1]

vp = 3500 * ones(Float32,n)
vs = vp ./ sqrt(3f0)
phi = 0.25f0 * ones(Float32,n)

rho = 2200 * ones(Float32,n)

vp_stack = [(Patchy(sw_true[:,:,i]',vp,vs,rho,phi))[1] for i = 1:nv]

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
zsrc_stack = [convertToCell(ContJitter((n[2]-1)*d[2], nsrc)) for i = 1:nv]

xrec = range((n[1]-1)*d[1],stop=(n[1]-1)*d[1], length=nrec)
yrec = 0f0
zrec = range(d[2],stop=(n[2]-1)*d[2],length=nrec)

srcGeometry_stack = [Geometry(xsrc, ysrc, zsrc_stack[i]; dt=dtS, t=timeS) for i = 1:nv]
recGeometry = Geometry(xrec, yrec, zrec; dt=dtR, t=timeR, nsrc=nsrc)

f0 = 0.02f0     # kHz
wavelet = ricker_wavelet(timeS, dtS, f0)
q_stack = [judiVector(srcGeometry_stack[i], wavelet) for i = 1:nv]

ntComp = get_computational_nt(srcGeometry_stack[end], recGeometry, model[end])
info = Info(prod(n), nsrc, ntComp)

opt = Options(return_array=true)
Pr = judiProjection(info, recGeometry)
Ps_stack = [judiProjection(info, srcGeometry_stack[i]) for i = 1:nv]

F = [Pr*judiModeling(info, model[i]; options=opt)*Ps_stack[i]' for i = 1:nv]

try
    JLD2.@load "data/data/time_lapse_data_$(nv)nv_$(nsrc)nsrc_jitter_sc.jld2" d_obs
    global d_obs = d_obs
    println("found data, loading")
catch e
    println("generating")
    global d_obs = [F[i]*q_stack[i] for i = 1:nv]
    JLD2.@save "data/data/time_lapse_data_$(nv)nv_$(nsrc)nsrc_jitter_sc.jld2" d_obs
end

λ = 0f0 # 2 norm regularization

x_true_code = x_test[:,:,1,1,16]
x_true = decode(x_test[:,:,1,1,16])
mutable struct compgrid
    d :: Tuple
    n :: Tuple
end

comp_grid = compgrid(d,n)

options          = PARSDMM_options()
options.FL       = Float32
options.feas_tol = 0.001f0
options.evol_rel_tol = 0.0001f0
set_zero_subnormals(true)

constraint = Vector{SetIntersectionProjection.set_definitions}()

# TV:
TV = get_TD_operator(comp_grid,"TV",options.FL)[1]
m_min     = 0.0
m_max     = 1.2f0*norm(TV*vec(x_true_code),1)
set_type  = "l1"
TD_OP     = "TV"
app_mode  = ("matrix","")
custom_TD_OP = ([],false)
push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

BLAS.set_num_threads(12)
(P_sub,TD_OPS,set_Prop) = setup_constraints(constraint,comp_grid,options.FL)
(TD_OPS,AtA,l,y) = PARSDMM_precompute_distribute(TD_OPS,set_Prop,comp_grid,options)

function prj(x; vmin=10f0, vmax=130f0)
    @time (x_perm1,log_PARSDMM) = PARSDMM(vec(x),AtA,TD_OPS,set_Prop,P_sub,comp_grid,options);
    xtv = reshape(x_perm1,n)
    y = decode(xtv)
    z = max.(min.(y,vmax),vmin)
    return encode(z)
end

#x = encode(x_normalizer,20f0*ones(Float32,nx,ny))[:,:,1]
x = zeros(Float32, nx, ny)
x_init = decode(x)

fig, ax = subplots(nrows=1,ncols=1,figsize=(20,12))
figloss, axloss = subplots(nrows=1,ncols=1,figsize=(20,12));axloss.set_title("loss");
figmisfit, axmisfit = subplots(nrows=1,ncols=1,figsize=(20,12));axmisfit.set_title("misfit");
figprior, axprior = subplots(nrows=1,ncols=1,figsize=(20,12));axprior.set_title("prior");

hisloss = zeros(Float32, grad_iterations)
hismisfit = zeros(Float32, grad_iterations)
hisprior = zeros(Float32, grad_iterations)
nssample = 4

ls = BackTracking(c_1=1f-4,iterations=10,maxstep=Inf32,order=3,ρ_hi=5f-1,ρ_lo=1f-1)
fval = Inf32
misfit = Inf32
prior = Inf32
α = 5f0


for j=1:grad_iterations

    println("Iteration ", j)
    rand_ns = [jitter(nsrc, nssample) for i = 1:nv]
    G_stack = [Forward(F[i][rand_ns[i]],q_stack[i][rand_ns[i]]) for i = 1:nv]
    d_obs_sample = [sample_src(d_obs[i], nsrc, rand_ns[i]) for i = 1:nv]

    function f(x)
        sw = decode(y_normalizer,NN(perm_to_tensor(x,nt,grid,dt)))[:,:,survey_indices,1]
        vp_stack = [(Patchy(sw[:,:,i]',vp,vs,rho,phi))[1] for i = 1:nv]
        d_predict = [G_stack[i]((1f3 ./ vp_stack[i]).^2f0) for i = 1:nv]
        global misfit = 0.5f0/nssample/nv * norm(d_predict-d_obs_sample)^2f0
        global prior = 0f0
        global fval = misfit + prior
        @show fval, misfit, prior
        return fval
    end

    θ = Flux.params(x)
    @time grads = gradient(θ) do
        return f(x)
    end
    gvec = grads.grads[x]::Matrix{Float32}
    p = -gvec/norm(gvec, Inf)

    # linesearch
    function ϕ(α)::Float32
        try
            global fval = f(prj(x + α * p))
        catch e
            @assert typeof(e) == DomainError
            global fval = Float32(Inf)
        end
        @show α, fval
        return fval
    end

    global α, fval = ls(ϕ, α, fval, dot(gvec, p))

    hisloss[j] = fval
    hismisfit[j] = misfit
    hisprior[j] = prior

    # Update model and bound projection
    global x = prj(x .+ α .* p)::Matrix{Float32}
    plot_velocity(decode(x), d; vmin=10f0, vmax=130f0, ax=ax, new_fig=false, name="inversion after $j iterations");
    axloss.plot(hisloss[1:j])
    axmisfit.plot(hismisfit[1:j])
    axprior.plot(hisprior[1:j])

    fig.savefig("result/scjittertvperm$(j).png", bbox_inches="tight",dpi=300)
    figloss.savefig("result/scjittertvloss$(j).png", bbox_inches="tight",dpi=300)
    figmisfit.savefig("result/scjittertvmisfit$(j).png", bbox_inches="tight",dpi=300)
    figprior.savefig("result/scjittertvprior$(j).png", bbox_inches="tight",dpi=300)

    println("Coupled inversion iteration no: ",j,"; function value: ",fval)
    JLD2.@save "result/scjittertvcoupleinversioniter$(j).jld2" x fval gvec hisloss hismisfit hisprior
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

