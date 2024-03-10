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
matplotlib.use("Agg")

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

## network structure
batch_size = 12
learning_rate = 2f-3
epochs = 5000
modes = 24
width = 32

AN = ActNorm(ntrain)
AN.forward(reshape(models, n[1], n[2], 1, ntrain));

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

NN = Net2d(modes, width; in_channels=4, out_channels=1, mid_channels=128)
gpu_flag && (global NN = NN |> gpu)

Flux.trainmode!(NN, true)
w = Flux.params(NN)

opt = Flux.Optimise.ADAMW(learning_rate, (0.9f0, 0.999f0), 1f-4)
nbatches = Int(floor(ntrain/batch_size))

Loss = zeros(Float32,epochs*nbatches)
Loss_valid = zeros(Float32, epochs)
prog = Progress(ntrain * epochs)

# plot figure
x_plot = x_valid[:, :, 1]
y_plot = y_valid[:, :, 1, 1]

# Define result directory

sim_name = "2D_FNO_vc"
exp_name = "velocity-continuation"

save_dict = @strdict exp_name
plot_path = plotsdir(sim_name, savename(save_dict; digits=6))

## training

for ep = 1:epochs

    Base.flush(Base.stdout)
    idx_e = reshape(randperm(ntrain)[1:batch_size*nbatches], batch_size, nbatches)

    Flux.trainmode!(NN, true)
    for b = 1:nbatches
        x = tensorize(x_train[:, :, idx_e[:,b]], grid, AN)
        y = y_train[:, :, idx_e[:,b]]
        if gpu_flag
            x = x |> gpu
            y = y |> gpu
        end
        println(size(x), size(y))
        grads = gradient(w) do
            global loss = norm(NN(x)-y)^2f0
            return loss
        end
        Loss[(ep-1)*nbatches+b] = loss
        for p in w
            Flux.Optimise.update!(opt, p, grads[p])
        end
        ProgressMeter.next!(prog; showvalues = [(:loss, loss), (:epoch, ep), (:batch, b)])
    end

    (ep % 100 !== 0) && continue

    Flux.testmode!(NN, true)
    y_predict = NN(tensorize(x_plot, grid, AN) |> gpu)   |> cpu

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
    fig_name = @strdict ep batch_size Loss modes width learning_rate epochs n d AN ntrain nvalid nsamples
    safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_2Dfno_vc.png"), fig);
    close(fig)

    Loss_valid[ep] = norm((NN(tensorize(x_valid, grid, AN) |> gpu)) - (y_valid |> gpu))^2f0 * batch_size/nvalid

    loss_train = Loss[1:ep*nbatches]
    loss_valid = Loss_valid[1:ep]
    fig = figure(figsize=(20, 12))
    subplot(1,3,1)
    plot(loss_train)
    title("training loss at epoch $ep")
    subplot(1,3,2)
    plot(1:nbatches:nbatches*ep, loss_valid); 
    title("validation loss at epoch $ep")
    subplot(1,3,3)
    plot(loss_train);
    plot(1:nbatches:nbatches*ep, loss_valid); 
    xlabel("iterations")
    ylabel("value")
    title("Objective function at epoch $ep")
    legend(["training", "validation"])
    tight_layout();
    safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_3Dfno_loss.png"), fig);
    close(fig);

    # NN_save = NN |> cpu
    # w_save = Flux.params(NN_save)    

    # param_dict = @strdict ep NN_save w_save batch_size Loss modes width learning_rate epochs n d AN ntrain nvalid loss_train loss_valid nsamples
    # @tagsave(
    #     datadir(sim_name, savename(param_dict, "jld2"; digits=6)),
    #     param_dict;
    #     safe=true
    # )
    
end

# NN_save = NN |> cpu
# w_save = params(NN_save)

# final_dict = @strdict Loss Loss_valid epochs NN_save w_save batch_size Loss modes width learning_rate epochs n d AN ntrain nvalid nsamples

# @tagsave(
#     datadir(sim_name, savename(final_dict, "jld2"; digits=6)),
#     final_dict;
#     safe=true
# )
