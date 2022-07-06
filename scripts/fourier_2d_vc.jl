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
matplotlib.use("Agg")

Random.seed!(1234)

# Define raw data directory
mkpath(datadir("training-data"))
input_path = datadir("training-data", "training-pairs.h5")

# Download the dataset into the data directory if it does not exist
if ~isfile(input_path)
    run(`wget https://www.dropbox.com/s/yj8n35qglol66db/'
        'training-pairs.h5 -q -O $input_path`)
end

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
batch_size = 5
learning_rate = 2f-3
epochs = 500
modes = 24
width = 32

AN = ActNorm(ntrain)
AN.forward(reshape(models, n[1], n[2], 1, nsamples)[:,:,:,1:ntrain]);

y_train = reshape(images, n[1], n[2], 1, nsamples)[:,:,:,1:ntrain];
y_valid = reshape(images, n[1], n[2], 1, nsamples)[:,:,:,ntrain+1:ntrain+nvalid];

function tensorize(x::AbstractMatrix{Float32},grid::Array{Float32,3},AN::ActNorm)
    # input nx*ny, output nx*ny*4*1
    nx, ny, _ = size(grid)
    return cat(reshape(AN(reshape(x, nx, ny, 1, 1))[:,:,1,1], nx, ny, 1, 1),
    reshape(image_base, nx, ny, 1, 1), reshape(grid, nx, ny, 2, 1), dims=3)
end

tensorize(x::AbstractArray{Float32,3},grid::Array{Float32,3},AN::ActNorm) = cat([tensorize(x[:,:,i],grid,AN) for i = 1:size(x,3)]..., dims=4)

x_train = tensorize(models[1:s:end,1:s:end,1:ntrain],grid,AN);
x_valid = tensorize(models[1:s:end,1:s:end,ntrain+1:ntrain+nvalid],grid,AN);

NN = Net2d(modes, width; in_channels=4, out_channels=1, mid_channels=128)
gpu_flag && (global NN = NN |> gpu)

Flux.trainmode!(NN, true)
w = Flux.params(NN)

opt = Flux.Optimise.ADAMW(learning_rate, (0.9f0, 0.999f0), 1f-4)
nbatches = Int(ntrain/batch_size)

Loss = zeros(Float32,epochs*nbatches)
Loss_valid = zeros(Float32, epochs)
prog = Progress(ntrain * epochs)

# plot figure
x_plot = x_valid[:, :, :, 1:1]
y_plot = y_valid[:, :, 1, 1]

# Define result directory

sim_name = "2D_FNO_vc"
exp_name = "velocity-continuation"

save_dict = @strdict exp_name
plot_path = plotsdir(sim_name, savename(save_dict; digits=6))

## training

for ep = 1:epochs

    Base.flush(Base.stdout)
    idx_e = reshape(randperm(ntrain), batch_size, nbatches)

    Flux.trainmode!(NN, true)
    for b = 1:nbatches
        x = x_train[:, :, :, idx_e[:,b]]
        y = y_train[:, :, idx_e[:,b]]
        if gpu_flag
            x = x |> gpu
            y = y |> gpu
        end
        grads = gradient(w) do
            global loss = .5f0 * norm(NN(x)-y)^2f0
            return loss
        end
        Loss[(ep-1)*nbatches+b] = loss
        for p in w
            Flux.Optimise.update!(opt, p, grads[p])
        end
        ProgressMeter.next!(prog; showvalues = [(:loss, loss), (:epoch, ep), (:batch, b)])
    end

    Flux.testmode!(NN, true)
    y_predict = NN(x_plot |> gpu)   |> cpu

    fig = figure(figsize=(20, 12))

    subplot(3,2,1)
    imshow(x_plot[:,:,1,1])
    title("background model")

    subplot(3,2,2)
    imshow(x_plot[:,:,2,1])
    title("given RTM")

    subplot(3,2,3)
    imshow(y_predict[:,:,1], vmin=minimum(y_plot), vmax=maximum(y_plot))
    title("predicted continuted RTM")

    subplot(3,2,4)
    imshow(y_plot, vmin=minimum(y_plot), vmax=maximum(y_plot))
    title("true continuted RTM")

    subplot(3,2,5)
    imshow(5f0 .* abs.(y_predict[:,:,1]-y_plot), vmin=minimum(y_plot), vmax=maximum(y_plot))
    title("5X abs difference")

    tight_layout()
    fig_name = @strdict ep batch_size Loss modes width learning_rate epochs n d AN ntrain nvalid nsamples
    safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_2Dfno_vc.png"), fig);
    close(fig)

    valid_idx = randperm(nvalid)[1:batch_size]
    Loss_valid[ep] = .5f0 * norm((NN(x_valid[:, :, :, valid_idx] |> gpu)) - (y_valid[:, :, valid_idx] |> gpu))^2f0

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

    NN_save = NN |> cpu
    w_save = params(NN_save)    

    param_dict = @strdict ep NN_save w_save batch_size Loss modes width learning_rate epochs n d AN ntrain nvalid loss_train loss_valid nsamples
    @tagsave(
        datadir(sim_name, savename(param_dict, "jld2"; digits=6)),
        param_dict;
        safe=true
    )
    
end

NN_save = NN |> cpu
w_save = params(NN_save)

final_dict = @strdict Loss Loss_valid epochs NN_save w_save batch_size Loss modes width learning_rate epochs n d AN ntrain nvalid nsamples

@tagsave(
    datadir(sim_name, savename(final_dict, "jld2"; digits=6)),
    final_dict;
    safe=true
)
