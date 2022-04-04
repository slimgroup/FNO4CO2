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

ntrain = 1000
ntest = 100

# load the network
BSON.@load "data/TrainedNet/2phasenet_200.bson" NN w batch_size Loss modes width learning_rate epochs gamma step_size

n = (64,64) # spatial dimension
d = 1f0./n  # normalized spacing
nx, ny = n
dx, dy = d

nt = 51     # num of time steps
dt = 1f0/nt # normalized time stepping

# load training pairs, set normalization for input and output
x_train, x_test, y_train, y_test, x_normalizer, y_normalizer, grid = loadPairs();
std_, eps_, mean_ = setMeanStd(x_normalizer)

# observation vintages
nv = 11
survey_indices = Int.(round.(range(1, stop=22, length=nv)))
sw_true = y_test[:,:,survey_indices,1]  # take true saturation from the 1st sample in the test set

# value, x, y, t
Flux.testmode!(NN, true);
Flux.testmode!(NN.conv1.bn0);
Flux.testmode!(NN.conv1.bn1);
Flux.testmode!(NN.conv1.bn2);
Flux.testmode!(NN.conv1.bn3);

# set up rock physics

vp = 3500 * ones(Float32,n)     # p-wave
vs = vp ./ sqrt(3f0)            # s-wave
phi = 0.25f0 * ones(Float32,n)  # porosity
rho = 2200 * ones(Float32,n)    # density
vp_stack = [(Patchy(sw_true[:,:,i]',vp,vs,rho,phi))[1] for i = 1:nv]    # time-varying vp

##### Wave equation

d = (15f0, 15f0)        # discretization for wave equation
o = (0f0, 0f0)          # origin

extentx = (n[1]-1)*d[1] # width of model
extentz = (n[2]-1)*d[2] # depth of model

nsrc = 32       # num of sources
nrec = 960      # num of receivers

model = [Model(n, d, o, (1f3 ./ vp_stack[i]).^2f0; nb = 80) for i = 1:nv]   # wave model

timeS = timeR = 750f0               # recording time
dtS = dtR = 1f0                     # recording time sampling rate
ntS = Int(floor(timeS/dtS))+1       # time samples
ntR = Int(floor(timeR/dtR))+1       # source time samples

# source locations -- on the left hand side of the model
xsrc = convertToCell(range(1*d[1],stop=1*d[1],length=nsrc))
ysrc = convertToCell(range(0f0,stop=0f0,length=nsrc))
zsrc = convertToCell(range(d[2],stop=extentz,length=nsrc))

# receiver locations -- on the right hand side of the model
xrec = range(extentx,stop=extentx,length=nrec)
yrec = 0f0
zrec = range(d[2],stop=extentz,length=nrec)

# set up src/rec geometry
srcGeometry = Geometry(xsrc, ysrc, zsrc; dt=dtS, t=timeS)
recGeometry = Geometry(xrec, yrec, zrec; dt=dtR, t=timeR, nsrc=nsrc)

# set up source
f0 = 0.02f0     # kHz
wavelet = ricker_wavelet(timeS, dtS, f0)
q = judiVector(srcGeometry, wavelet)

# set up computational time step (used for simulation)
ntComp = get_computational_nt(srcGeometry, recGeometry, model[end])
info = Info(prod(n), nsrc, ntComp)
opt = Options(return_array=true)

# set up simulation operators
Pr = judiProjection(info, recGeometry)  # receiver restriction
Ps = judiProjection(info, srcGeometry)  # source injection
Ainv = [judiModeling(info, model[i]; options=opt) for i = 1:nv]  # acoustic wave equation solver
Ftrue = [Pr*Ainv[i]*Ps' for i = 1:nv]  # forward modeling operator

## generate/load data
try
    JLD2.@load "data/data/time_lapse_data_$(nv)vintage_$(nsrc)nsrc.jld2" d_obs
    global d_obs = d_obs
    println("found data, loading")
catch e
    println("generating")
    global d_obs = [Ftrue[i]*q for i = 1:nv]
    JLD2.@save "data/data/time_lapse_data_$(nv)vintage_$(nsrc)nsrc.jld2" d_obs
end

# take the 1st sample from the test dataset
x_true = decode(x_test[:,:,1,1,1])  # take a test sample

# initial x and its code (by normalization)
x = zeros(Float32, nx, ny)
x_init = decode(x)

# set up plots
grad_iterations = 120
fig, ax = subplots(nrows=1,ncols=1,figsize=(20,12))
figloss, axloss = subplots(nrows=1,ncols=1,figsize=(20,12));axloss.set_title("loss");
hisloss = zeros(Float32, grad_iterations)

# batchsize in wave equation, i.e. in each iteration the number of sources for each vintage to compute the gradient
nssample = 8

# BackTracking linesearch algorithm
ls = BackTracking(c_1=1f-4,iterations=20,maxstep=Inf32,order=3,ρ_hi=5f-1,ρ_lo=1f-1)
fval = Inf32
α = 5f0

# set up projection operator from SetIntersectionProjection
prj = SetPrj()

for j=1:grad_iterations

    println("Iteration ", j)
    rand_ns = [jitter(nsrc, nssample) for i = 1:nv]                             # select random sources for each vintage
    F = [Forward(Ftrue[i][rand_ns[i]],q[rand_ns[i]]) for i = 1:nv]              # set-up wave modeling operator (custom adjoint is from JUDI4Flux)
    dobs = [sample_src(d_obs[i], nsrc, rand_ns[i]) for i = 1:nv]                # subsampled seismic dataset from the selected sources

    # function value
    function f(K)
        println("evaluate f")
        c = S(K); v = R(c); dpred = F(v);
        global loss = 0.5f0 * norm(dpred.-dobs)^2f0
        @show loss
        return loss
    end

    θ = Flux.params(x)
    @time grads = gradient(θ) do
        return f(x)
    end

    # get the gradient from AD
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

    hisloss[j] = loss

    # Update model and bound projection
    global x = prj(x + α * p)::Matrix{Float32}
    plot_velocity(decode(x), d; vmin=10f0, vmax=130f0, ax=ax, new_fig=false, name="inversion after $j iterations");
    axloss.plot(hisloss[1:j])

    fig.savefig("result/perm$(j).png", bbox_inches="tight",dpi=300)
    figloss.savefig("result/loss$(j).png", bbox_inches="tight",dpi=300)

    println("Coupled inversion iteration no: ",j,"; function value: ",loss)
    JLD2.@save "result/coupleinversioniter$(j).jld2" x fval gvec hisloss
end

# plot the results
figure(figsize=(20,12));
subplot(1,3,1)
imshow(x_init,vmin=20,vmax=120);title("initial permeability");
subplot(1,3,2);
imshow(decode(x),vmin=20,vmax=120);title("inversion by NN, $(grad_iterations) iter");
subplot(1,3,3);
imshow(x_true,vmin=20,vmax=120);title("GT permeability");

# plot loss
figure();
plot(hisloss)

