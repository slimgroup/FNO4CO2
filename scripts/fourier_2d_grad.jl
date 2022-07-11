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

### save path
sim_name = "2DFNOinversion"
exp_name = "velocity-continuation"

save_dict = @strdict exp_name
plot_path = plotsdir(sim_name, savename(save_dict; digits=6))

# Define raw data directory
mkpath(datadir("training-data"))
input_path = datadir("training-data", "training-pairs.h5")

# Download the dataset into the data directory if it does not exist
if ~isfile(input_path)
    run(`wget https://www.dropbox.com/s/yj8n35qglol66db/'
        'training-pairs.h5 -q -O $input_path`)
end

# load network
JLD2.@load "../data/2D_FNO_vc/batch_size=5_ep=500_epochs=500_learning_rate=0.002_modes=24_nsamples=205_ntrain=200_nvalid=5_width=32.jld2"
NN = deepcopy(NN_save);
Flux.testmode!(NN, true);

# load data
input = h5open(input_path, "r");

## d, n
n = (160, 205)
d = 1f0 ./ n
nsamples = 205
ntrain = 200
nvalid = 5
grid = gen_grid(n, d);

# load each entry
image_base = read(input["image-base"])[:,:,1];
images = read(input["images"]);
model_base = read(input["model-base"])[:,:,1];
models = read(input["models"]);

## network structure
batch_size = 10
learning_rate = 2f-3
epochs = 500
modes = 24
width = 32

y_train = reshape(images, n[1], n[2], 1, nsamples)[:,:,:,1:ntrain];
y_valid = reshape(images, n[1], n[2], 1, nsamples)[:,:,:,ntrain+1:ntrain+nvalid];

function tensorize(x::AbstractMatrix{Float32},grid::Array{Float32,3},AN::ActNorm)
    # input nx*ny, output nx*ny*4*1
    nx, ny, _ = size(grid)
    return cat(reshape(AN(reshape(x, nx, ny, 1, 1))[:,:,1,1], nx, ny, 1, 1),
    reshape(image_base, nx, ny, 1, 1), reshape(grid, nx, ny, 2, 1), dims=3)
end

tensorize(x::AbstractArray{Float32,3},grid::Array{Float32,3},AN::ActNorm) = cat([tensorize(x[:,:,i],grid,AN) for i = 1:size(x,3)]..., dims=4)

x_train = tensorize(models[:,:,1:ntrain],grid,AN);
x_valid = tensorize(models[:,:,ntrain+1:ntrain+nvalid],grid,AN);

# plot figure
x_plot = models[:,:,ntrain+1]
y_plot = y_valid[:, :, 1, 1]

@time y_predict = NN(x_valid[:, :, :, 1:1]);

d = (1f1, 2.5f1)
fig = figure(figsize=(16, 12))

subplot(3,2,1)
plot_velocity(x_plot[:,:,1,1], (1f1, 2.5f1); new_fig=false, vmin=0, vmax=0.2, name="background model", cmap="GnBu"); colorbar();

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

x_water = mean(models[:,:,1:ntrain], dims=3)[:,:,1];
for i = 1:n[2]
    x_water[:,i] .= x_water[:,100]
end
x = deepcopy(x_water);
x_init = deepcopy(x);
# function value
function f(x)
    println("evaluate f")
    loss = 0.5f0 * norm(NN(tensorize(x, grid, AN))-y_plot)^2f0
    return loss
end

# set up plots
niterations = 1000

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
        x1 = x .+ α .* gnorm
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
    global α = 1.2f0 * step
    hisloss[j+1] = fval

    # Update model and bound projection
    global x .= x .+ step .* gnorm

    ProgressMeter.next!(prog; showvalues = [(:loss, fval), (:iter, j), (:steplength, step)])

end

y_inv = NN(tensorize(x, grid, AN))
y_init = NN(tensorize(x_init, grid, AN))

d = (1f1, 2.5f1)
fig = figure(figsize=(20, 12))

subplot(3,3,1)
plot_velocity(x_plot, (1f1, 2.5f1); new_fig=false, vmin=0, vmax=0.2, name="true background model", cmap="GnBu"); colorbar();

subplot(3,3,2)
plot_velocity(x, (1f1, 2.5f1); new_fig=false, vmin=0, vmax=0.2, name="inverted background model", cmap="GnBu"); colorbar();

subplot(3,3,3)
plot_velocity(x_init, (1f1, 2.5f1); new_fig=false, vmin=0, vmax=0.2, name="initial background model", cmap="GnBu"); colorbar();

subplot(3,3,4)
plot_simage(y_plot, (1f1, 2.5f1); new_fig=false, cmap="seismic", vmax=1f2, name="true continued RTM"); colorbar();

subplot(3,3,5)
plot_simage(y_inv[:,:,1], (1f1, 2.5f1); new_fig=false, cmap="seismic", vmax=1f2, name="FNO(inverted background)"); colorbar();

subplot(3,3,6)
plot_simage(y_init[:,:,1], (1f1, 2.5f1); new_fig=false, cmap="seismic", vmax=1f2, name="FNO(initial background)"); colorbar();

subplot(3,3,7)
plot(y_inv[:,80,1]);
plot(y_plot[:,80]);
plot(y_init[:,80,1]);
legend(["inverted","true", "initial"])
title("RTM vertical profile at 2km")

subplot(3,3,8)
plot(x[:,80]);
plot(x_plot[:,80]);
plot(x_init[:,80]);
legend(["inverted","true", "initial"])
title("background vertical profile at 2km")
tight_layout()

fig_name = @strdict niterations α
safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_2Dfno_vc_inv.png"), fig);

## loss
fig = figure(figsize=(20,12));
plot(hisloss[1:niterations+1]);title("loss");
tight_layout()

fig_name = @strdict niterations α
safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_2Dfno_vc_loss.png"), fig);

