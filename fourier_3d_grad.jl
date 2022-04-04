# author: Ziyi Yin, ziyi.yin@gatech.edu 

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
using Optim: optimize, LBFGS, Options, only_fg!

CUDA.culiteral_pow(::typeof(^), a::Complex{Float32}, b::Val{2}) = real(conj(a)*a)
CUDA.sqrt(a::Complex) = cu(sqrt(a))
Base.broadcasted(::typeof(sqrt), a::Base.Broadcast.Broadcasted) = Base.broadcast(sqrt, Base.materialize(a))

include("utils.jl")
include("fno3dstruct.jl")
include("inversion_utils.jl");

Random.seed!(3)

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

# set test mode of the network
Flux.testmode!(NN, true)
Flux.testmode!(NN.conv1.bn0);
Flux.testmode!(NN.conv1.bn1);
Flux.testmode!(NN.conv1.bn2);
Flux.testmode!(NN.conv1.bn3);

# take a test sample
x_true = decode(x_test[:,:,1,1,1])  # take a test sample

# observation vintages
nv = nt
survey_indices = Int.(round.(range(1, stop=nt, length=nv)))
yobs = y_test[:,:,survey_indices,1] # ground truth CO2 concentration at these vintages

# initial x and its code (by normalization)
x = zeros(Float32, nx, ny)
x_init = decode(x_normalizer,reshape(x,nx,ny,1))[:,:,1]

# function value
function f(x)
    println("evaluate f")
    loss = 0.5f0 * norm(S(prj(x))-yobs)^2f0
    println("fval=", loss)
    return loss
end

# function value and gradient
function fg!(fval,g,x)
    println("evaluate f and g")
    p = Flux.params(x)
    @time grads = gradient(p) do
        global loss = f(x)
        return loss
    end
    copyto!(g, grads.grads[x])
    return loss
end

# what to do after every iteration
function callb(os)
    x = os.metadata["x"] #get current optimization variable (latent x)
    fval[curr_iter] = f(x)
    println("iter: "*string(curr_iter)*" fval: "*string(fval[curr_iter]))
    flush(stdout)
    # show the result in every iteration
    axloss.plot(fval[1:curr_iter])
    ax.imshow(decode(x), vmin=20, vmax=120)
    global curr_iter += 1

    #if you want the callback to stop the procedure under some conditions, return true
    return false
end

# bound constraint
function prj(x; vmin=10f0, vmax=130f0)
    y = decode(x)
    z = max.(min.(y,vmax),vmin)
    return encode(z)
end

# set up plots
curr_iter = 1
_, ax = subplots(nrows=1, ncols=1, figsize=(20,12))
_, axloss = subplots(nrows=1, ncols=1, figsize=(20,12))

## LBFGS by fg function
lbfgs_iters = 50
fval = zeros(Float32, lbfgs_iters+1)
res = optimize(only_fg!(fg!), x, LBFGS(), Options(iterations=lbfgs_iters, extended_trace=true, callback=callb))

## compute true and plot
figure(figsize=(20,12));
subplot(1,3,1)
imshow(x_init,vmin=20,vmax=120);title("initial permeability");
subplot(1,3,2);
imshow(decode(res.minimizer),vmin=20,vmax=120);title("inversion by NN, $(grad_iterations) iter");
subplot(1,3,3);
imshow(x_true,vmin=20,vmax=120);title("GT permeability");
