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

####### NEW STUFF DFNO ###########
using DFNO:DFNO_3D
using MPI

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
pe_count = MPI.Comm_size(comm)

# CUDA.device!(rank % 4)

partition = [1,pe_count]

####### END NEW STUFF DFNO ###########

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

####### NEW STUFF DFNO ###########

modelConfig = DFNO_3D.ModelConfig(nc_in=4, nc_lift=width, nc_out=1, nx=1, ny=n[2], nz=1, nt=n[1], mx=1, my=modes*2, mz=1, mt=modes, nblocks=4, partition=partition, dtype=Float32, relu01=false)
model = DFNO_3D.Model(modelConfig)
θ = DFNO_3D.initModel(model)

filename = "./data/2D_FNO_vc/batch_size=12_ep=3700_epochs=5000_learning_rate=0.002_modes=24_nsamples=210_ntrain=205_nvalid=5_width=32.jld2"
DFNO_3D.loadWeights!(θ, filename, "θ_save", partition, isLocal=false)

####### END NEW STUFF DFNO ###########

opt = Flux.Optimise.ADAMW(learning_rate, (0.9f0, 0.999f0), 1f-4)
nbatches = Int(floor(ntrain/batch_size))

Loss = zeros(Float32,epochs*nbatches)
Loss_valid = zeros(Float32, epochs)
prog = Progress(ntrain * epochs)

# Define result directory

sim_name = "2D_FNO_vc"
exp_name = "velocity-continuation-validation"

save_dict = @strdict exp_name
plot_path = plotsdir(sim_name, savename(save_dict; digits=6))

n_columns = nvalid + 1

fig = figure(figsize=(16 * n_columns, 20))

# Plot for training sample
x_temp = tensorize(x_train[:, :, 10], grid, AN) |> gpu
x_temp = permutedims(x_temp, [3, 1, 2, 4])
y_temp_train = reshape(DFNO_3D.forward(model, θ, x_temp), n) |> cpu

subplot(3, n_columns, 1)
plot_simage(y_train[:,:,10], (1f1, 2.5f1); new_fig=false, cmap="seismic", vmax=1f2, name="true continued RTM (training)"); colorbar();
subplot(3, n_columns, n_columns + 1)
plot_simage(y_temp_train[:,:,1], (1f1, 2.5f1); new_fig=false, cmap="seismic", vmax=1f2, name="predicted continued RTM (training)"); colorbar();
subplot(3, n_columns, 2 * n_columns + 1)
plot_simage(y_temp_train[:,:,1] - y_train[:,:,10], (1f1, 2.5f1); new_fig=false, cmap="RdGy", vmax=2f1, name="diff (training)"); colorbar();

# Loop for validation samples
for i in 1:nvalid
    # True data
    y_plot = y_valid[:, :, i]
    subplot(3, n_columns, i + 1)
    plot_simage(y_plot, (1f1, 2.5f1); new_fig=false, cmap="seismic", vmax=1f2, name="true continued RTM (validation $i)"); colorbar();

    # Predicted data
    x_temp = tensorize(x_valid[:, :, i], grid, AN) |> gpu
    x_temp = permutedims(x_temp, [3, 1, 2, 4])
    y_predict = reshape(DFNO_3D.forward(model, θ, x_temp), n) |> cpu
    subplot(3, n_columns, n_columns + i + 1)
    plot_simage(y_predict[:,:,1], (1f1, 2.5f1); new_fig=false, cmap="seismic", vmax=1f2, name="predicted continued RTM (validation $i)"); colorbar();

    # Difference
    subplot(3, n_columns, 2 * n_columns + i + 1)
    plot_simage(y_predict[:,:,1] - y_plot, (1f1, 2.5f1); new_fig=false, cmap="RdGy", vmax=2f1, name="diff (validation $i)"); colorbar();
end

tight_layout()
fig_name = @strdict batch_size Loss modes width learning_rate epochs n d AN ntrain nvalid nsamples
safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_2Dfno_vc_extended.png"), fig);
close(fig)

MPI.Finalize()
