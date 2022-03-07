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
include("inversion_utils.jl");

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

nv = 5
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

nsrc = 16
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

grad_iterations = 80

function sample_src(d_obs, nsrc, rand_ns)
    datalength = Int(length(d_obs)/nsrc)
    return vcat([d_obs[(rand_ns[i]-1)*datalength+1:rand_ns[i]*datalength] for i = 1:length(rand_ns)]...)
end

function fg!(gvec, x_inv; nvsample = 5, nssample=4)
    println("evaluate f and g")
    p = params(x_inv)
    rand_nv = randperm(nv)[1:nvsample]
    rand_ns = [randperm(nsrc)[1:nssample] for i = 1:nvsample]
    d_obs_sample = [sample_src(d_obs[rand_nv[i]], nsrc, rand_ns[i]) for i = 1:nvsample]
    @time grads = gradient(p) do
        sw = decode(y_normalizer,NN(perm_to_tensor(x_inv,nt,grid,dt)))
        vp_stack = [(Patchy(sw[:,:,survey_indices[rand_nv[i]],1]',vp,vs,rho,phi))[1] for i = 1:nvsample]
        m_stack = [(1000f0 ./ vp_stack[i]).^2f0 for i = 1:nvsample]
        G_stack = [Forward(F[rand_nv[i]][rand_ns[i]],q[rand_ns[i]]) for i = 1:nvsample]
        d_predict = [G_stack[i](m_stack[i]) for i = 1:nvsample]
        global loss = 0.5f0 * norm(d_predict-d_obs_sample)^2f0
        return loss
    end
    copyto!(gvec, grads.grads[x_inv])
    return loss
end

function prj(x, vmin, vmax)
    x_perm = decode(x_normalizer,reshape(x,nx,ny,1))[:,:,1]
    x_perm1 = min.(max.(x_perm,vmin),vmax)
    return encode(x_normalizer,x_perm1)[:,:,1]
end

x = zeros(Float32, nx, ny)
x_init = decode(x_normalizer, reshape(x, nx, ny, 1))[:,:,1]

ls = BackTracking(c_1=1f-4,iterations=10,maxstep=Inf32,order=3,ρ_hi=5f-1,ρ_lo=1f-1)
Grad_Loss = zeros(Float32, grad_iterations)

T = Float32
vmin = 10f0
vmax = 130f0

#Grad_Loss[1] = f(x)
#println("Initial function value: ", Grad_Loss[1])

const ϵ = 1e-8
nvsample = 5
nssample = 2
v = zeros(Float32, nx, ny)

figure();

a = 0.5f0
b = 100f0
β = 0.99

for j=1:grad_iterations

    gvec = similar(x)::AbstractArray{T}
    fval = fg!(gvec, x; nssample=nssample, nvsample=nvsample)::T
    Grad_Loss[j] = fval
    η = Float32(a * (b+j)^(-1/3))
    @. v = β * v + (1-β) * gvec^2
    @. gvec *= η / (√v + ϵ)
    # Update model and bound projection
    global x = prj(x .- η.*gvec, vmin, vmax)::AbstractArray{T}

    global x_inv = decode(x_normalizer,reshape(x,nx,ny,1))[:,:,1]
    imshow(x_inv,vmin=20,vmax=120);title("inversion by NN, $j iter");
    println("Coupled inversion iteration no: ",j,"; function value: ",fval,"step length: ", η)

end

figure();
subplot(1,3,1)
imshow(x_init,vmin=20,vmax=120);title("initial permeability");
subplot(1,3,2);
imshow(x_inv,vmin=20,vmax=120);title("inversion by NN, $grad_iterations iter");
subplot(1,3,3);
imshow(x_test_1,vmin=20,vmax=120);title("GT permeability");

figure();
plot(Grad_Loss)

JLD2.@save "result/$(nv)nv_$(nsrc)nsrc_randomized.jld2" Grad_Loss x_inv