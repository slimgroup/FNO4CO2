# author: Ziyi Yin, ziyi.yin@gatech.edu
# This script trains a Fourier Neural Operator that does velocity continuation
# Details in IMAGE 2022 abstract @ https://arxiv.org/abs/2203.14386

using DrWatson
@quickactivate "FNO4CO2"

using FNO4CO2
using PyPlot
using JLD2
using Flux, Random, FFTW
using MAT, Statistics, LinearAlgebra
using CUDA
using ProgressMeter
using InvertibleNetworks:ActNorm
using HDF5
using SlimPlotting
using LineSearches
matplotlib.use("Agg")

Random.seed!(1234)

proj = false

# load network
JLD2.@load "../data/2D_FNO_vc/batch_size=12_ep=500_epochs=5000_learning_rate=0.002_modes=24_nsamples=210_ntrain=205_nvalid=5_width=32.jld2"
NN = deepcopy(NN_save);
Flux.testmode!(NN, true);

### save path
sim_name = "2DFNOinversion"
exp_name = "velocity-continuation"

save_dict = @strdict exp_name
plot_path = plotsdir(sim_name, savename(save_dict; digits=6))

Random.seed!(1234)

# Define raw data directory
mkpath(datadir("training-data"))
train_data_path = datadir("training-data", "training-pairs.h5")
test_data_path = datadir("training-data", "data-pairs.h5")

# Download the dataset into the data directory if it does not exist
if ~isfile(train_data_path)
    run(`wget https://www.dropbox.com/s/yj8n35qglol66db/'
        'training-pairs.h5 -q -O $train_data_path`)
end
if ~isfile(test_data_path)
    run(`wget https://www.dropbox.com/s/3c5siiqshwxjd6k/'
        'data-pairs.h5 -q -O $test_data_path`)
end

## d, n
n = (160, 205)
d = 1f0 ./ n
nsamples = 210
ntrain = 205
nvalid = 5
grid = gen_grid(n, d);

# load data
train_pairs = h5open(train_data_path, "r");
test_pairs = h5open(test_data_path, "r");

# load each entry
image_base = read(train_pairs["image-base"])[:,:,1];
images = read(train_pairs["images"]);
model_base = read(train_pairs["model-base"])[:,:,1];
models = read(train_pairs["models"]);

# load test/validation
images_test = read(test_pairs["images"]);
models_test = read(test_pairs["models"]);

y_train = images;
y_valid = images_test[:,:,1:nvalid];

function tensorize(x::AbstractMatrix{Float32},grid::Array{Float32,3},AN::ActNorm)
    # input nx*ny, output nx*ny*4*1
    nx, ny, _ = size(grid)
    return cat(reshape(AN(reshape(x, nx, ny, 1, 1))[:,:,1,1], nx, ny, 1, 1),
    reshape(image_base, nx, ny, 1, 1), reshape(grid, nx, ny, 2, 1), dims=3)
end

tensorize(x::AbstractArray{Float32,3},grid::Array{Float32,3},AN::ActNorm) = cat([tensorize(x[:,:,i],grid,AN) for i = 1:size(x,3)]..., dims=4)

x_train = models;
x_valid = models_test[:,:,1:nvalid];

# plot figure
x_plot = x_valid[:,:,1]
y_plot = y_valid[:,:,1]

@time y_predict = NN(tensorize(x_plot, grid, AN));

d = (1f1, 2.5f1)
fig = figure(figsize=(16, 12))

subplot(3,2,1)
plot_velocity(x_plot, (1f1, 2.5f1); new_fig=false, vmin=0, vmax=0.2, name="background model", cmap="GnBu"); colorbar();

subplot(3,2,2)
plot_simage(y_predict[:,:,1], (1f1, 2.5f1); new_fig=false, cmap="seismic", vmax=1f2, name="predicted continued RTM"); colorbar();

subplot(3,2,3)
plot_simage(y_plot, (1f1, 2.5f1); new_fig=false, cmap="seismic", vmax=1f2, name="true continued RTM"); colorbar();

subplot(3,2,4)
plot_simage(y_predict[:,:,1]-y_plot, (1f1, 2.5f1); new_fig=false, cmap="RdGy", vmax=2f1, name="diff"); colorbar();

subplot(3,2,5)
plot(y_predict[:,80,1]);
plot(y_plot[:,80]);
legend(["predict","true"])
title("vertical profile at 2km")

subplot(3,2,6)
plot(y_predict[:,164,1]);
plot(y_plot[:,164]);
legend(["predict","true"])
title("vertical profile at 4.12km")

tight_layout()
safesave(joinpath(plot_path, "_2Dfno_vc_predict.png"), fig);

## invert for background
y_obs = images_test[:,:,nvalid+1];
x_true = models_test[:,:,nvalid+1];

x_water = mean(x_train, dims=3)[:,:,1];
for i = 1:n[2]
    x_water[:,i] .= x_water[:,100]
end
x = deepcopy(x_water);
x_init = deepcopy(x);
# function value
function f(x)
    println("evaluate f")
    loss = 0.5f0 * norm(NN(tensorize(x, grid, AN))-y_obs)^2f0
    return loss
end

if proj
    # projection
    using SetIntersectionProjection
    ####### bound on vertical derivative #######

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

    #bounds:
    m_min     = -1f2
    m_max     = 0f0
    set_type  = "bounds"
    TD_OP     = "D_x"
    app_mode  = ("matrix","")
    custom_TD_OP = ([],false)
    push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP));

    BLAS.set_num_threads(12)
    (P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid,options.FL)
    (TD_OP,AtA,l,y) = PARSDMM_precompute_distribute(TD_OP,set_Prop,comp_grid,options)

    function prj(x::Matrix{Float32})
        @time (x1,log_PARSDMM) = PARSDMM(vec(x),AtA,TD_OP,set_Prop,P_sub,comp_grid,options);
        return reshape(x1, n)::Matrix{Float32}
    end
else
    # identity
    prj(x::Matrix{Float32}) = x::Matrix{Float32}
end

# set up plots
niterations = 500

hisloss = zeros(Float32, niterations+1)
ls = BackTracking(c_1=1f-4,iterations=10,maxstep=Inf32,order=3,ρ_hi=5f-1,ρ_lo=1f-1)
α = 1f1
### backtracking line search
prog = Progress(niterations)
for j=1:niterations

    p = Flux.params(x)

    @time grads = gradient(p) do
        global loss = f(x)
        println("evaluate g")
        return loss
    end
    (j==1) && (hisloss[1] = loss)
    g = grads.grads[x]
    gnorm = -g/norm(g, Inf)

    println("iteration no: ",j,"; function value: ",loss)

    # linesearch
    function ϕ(α)
        x1 = prj(x .+ α .* gnorm)
        misfit = f(x1)
        @show α, misfit
        return misfit
    end
    try
        global step, fval = ls(ϕ, α, loss, dot(g, gnorm))
    catch e
        println("linesearch failed at iteration: ",j)
        global niterations = j
        hisloss[j+1] = loss
        break
    end
    global α = 2f0 * step
    hisloss[j+1] = fval

    # Update model and bound projection
    global x .= prj(x .+ step .* gnorm)

    ProgressMeter.next!(prog; showvalues = [(:loss, fval), (:iter, j), (:steplength, step)])

end

y_inv = NN(tensorize(x, grid, AN))
y_init = NN(tensorize(x_init, grid, AN))

d = (1f1, 2.5f1)
fig = figure(figsize=(20, 12))

subplot(3,3,1)
plot_velocity(x_true, (1f1, 2.5f1); new_fig=false, vmin=0, vmax=0.2, name="true background model", cmap="GnBu"); colorbar();

subplot(3,3,2)
plot_velocity(x, (1f1, 2.5f1); new_fig=false, vmin=0, vmax=0.2, name="inverted background model", cmap="GnBu"); colorbar();

subplot(3,3,3)
plot_velocity(x_init, (1f1, 2.5f1); new_fig=false, vmin=0, vmax=0.2, name="initial background model", cmap="GnBu"); colorbar();

subplot(3,3,4)
plot_simage(y_obs, (1f1, 2.5f1); new_fig=false, cmap="seismic", vmax=1f2, name="true continued RTM"); colorbar();

subplot(3,3,5)
plot_simage(y_inv[:,:,1], (1f1, 2.5f1); new_fig=false, cmap="seismic", vmax=1f2, name="FNO(inverted background)"); colorbar();

subplot(3,3,6)
plot_simage(y_init[:,:,1], (1f1, 2.5f1); new_fig=false, cmap="seismic", vmax=1f2, name="FNO(initial background)"); colorbar();

subplot(3,3,7)
plot(y_inv[:,80,1]);
plot(y_obs[:,80]);
plot(y_init[:,80,1]);
legend(["inverted","true", "initial"])
title("RTM vertical profile at 2km")

subplot(3,3,8)
plot(x[:,80]);
plot(x_true[:,80]);
plot(x_init[:,80]);
legend(["inverted","true", "initial"])
title("background vertical profile at 2km")
tight_layout()

fig_name = @strdict proj niterations α
safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_2Dfno_vc_inv.png"), fig);

## loss
fig = figure(figsize=(20,12));
plot(hisloss[1:niterations+1]);title("loss");
tight_layout()

fig_name = @strdict proj niterations α
safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_2Dfno_vc_loss.png"), fig);

