using DrWatson
@quickactivate "FNO4CO2"

# using Pkg
# Pkg.instantiate()

include("../src/config.jl")

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
using DFNO:DFNO_3D
using CUDA
using MPI
using .Config

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
pe_count = MPI.Comm_size(comm)

params = Config.get_parameters()
offsets = params["n_offsets"]
partition = [1,pe_count]

## d, n
n = (512, 256)
d = 1f0 ./ n
nsamples = 512
ntrain = 500
nvalid = 10
grid = gen_grid(n, d);

## network structure
batch_size = 5
learning_rate = 2f-3
epochs = 5000
modes = 36
width = 32
offsets = 51

nc_in = offsets + 1 + 1 + 4 # offsets + 2 velocity models + indices
nc_out = offsets

@info "Initializing model..."

modelConfig = DFNO_3D.ModelConfig(nc_in=nc_in, nc_lift=width, nc_out=nc_out, nx=n[1], ny=n[2], nz=1, nt=1, partition=partition, dtype=Float32)

dataset_path = "/pscratch/sd/r/richardr/FNO-CIG/results/concatenated_data.jld2"
x_train, y_train, x_valid, y_valid = read_velocity_cigs_offsets_as_nc(dataset_path, modelConfig, ntrain=ntrain, nvalid=nvalid)

x_train = reshape(x_train, nc_in, n..., :)
x_valid = reshape(x_valid, nc_in, n..., :)
y_valid = reshape(y_valid, nc_out, n..., :)
y_train = reshape(y_train, nc_out, n..., :)

nc_in = offsets + 1 + 1 + 2 # remove z and t indices

x_train = x_train[1:nc_in, :, :, :]
x_valid = x_valid[1:nc_in, :, :, :]

modelConfig = DFNO_3D.ModelConfig(nc_in=nc_in, nc_lift=width, nc_out=nc_out, nx=n[1], ny=n[2], nz=1, nt=1, partition=partition, dtype=Float32)

x_train = permutedims(x_train, [2, 3, 1, 4])
x_valid = permutedims(x_valid, [2, 3, 1, 4])
y_valid = permutedims(y_valid, [2, 3, 1, 4])
y_train = permutedims(y_train, [2, 3, 1, 4])

@info "Loaded data..."

println(size(x_train), ":", size(y_train), ":", size(x_valid), ":", size(y_valid))

NN = Net2d(modes, width; in_channels=nc_in, out_channels=nc_out, mid_channels=128)
gpu_flag && (global NN = NN |> gpu)

Flux.trainmode!(NN, true)
w = Flux.params(NN)

opt = Flux.Optimise.ADAMW(learning_rate, (0.9f0, 0.999f0), 1f-4)
nbatches = Int(floor(ntrain/batch_size))

Loss = zeros(Float32,epochs*nbatches)
Loss_valid = zeros(Float32, epochs)
prog = Progress(ntrain * epochs)

# plot figure
x_plot = x_valid[:, :, :, 1:1]
y_plot = y_valid[:, :, :, 1:1]

# Define result directory

sim_name = "2D_FNO_vc"
exp_name = "velocity-continuation-cig-control"

save_dict = @strdict exp_name
plot_path = plotsdir(sim_name, savename(save_dict; digits=6))

## training

for ep = 1:epochs

    Base.flush(Base.stdout)
    idx_e = reshape(randperm(ntrain)[1:batch_size*nbatches], batch_size, nbatches)

    Flux.trainmode!(NN, true)
    for b = 1:nbatches
        x = x_train[:, :, :, idx_e[:,b]]
        y = y_train[:, :, :, idx_e[:,b]]
        if gpu_flag
            x = x |> gpu
            y = y |> gpu
        end
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

    Flux.testmode!(NN, true)
    Loss_valid[ep] = norm((NN(x_valid |> gpu)) - (y_valid |> gpu))^2f0 * batch_size/nvalid
    (ep % 100 !== 0) && continue

    y_predict = NN(x_plot |> gpu) |> cpu

    x_plot_temp = permutedims(x_plot, [3, 1, 2, 4])
    y_plot_temp = permutedims(y_plot, [3, 1, 2, 4])
    y_predict_plot = permutedims(y_predict, [3, 1, 2, 4])

    fig_name = @strdict ep batch_size Loss modes width learning_rate epochs n d ntrain nvalid nsamples
    plot_cig_eval(modelConfig, plot_path, fig_name, x_plot_temp, y_plot_temp, y_predict_plot)

    loss_train = Loss[1:ep*nbatches]
    loss_valid = Loss_valid[1:ep]

    PyPlot.rc("figure", titlesize=8)
    PyPlot.rc("font", family="serif"); PyPlot.rc("xtick", labelsize=8); PyPlot.rc("ytick", labelsize=8)
    PyPlot.rc("axes", labelsize=8)     # Default fontsize for x and y labels
    PyPlot.rc("axes", titlesize=8)     # Default fontsize for titles

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
    w_save = Flux.params(NN_save)    

    param_dict = @strdict ep NN_save w_save batch_size Loss modes width learning_rate epochs n d ntrain nvalid loss_train loss_valid nsamples
    @tagsave(
        datadir(sim_name, savename(param_dict, "jld2"; digits=6)),
        param_dict;
        safe=true
    )
end

NN_save = NN |> cpu
w_save = Flux.params(NN_save)

final_dict = @strdict Loss Loss_valid epochs NN_save w_save batch_size Loss modes width learning_rate epochs n d ntrain nvalid nsamples

@tagsave(
    datadir(sim_name, savename(final_dict, "jld2"; digits=6)),
    final_dict;
    safe=true
)

MPI.Finalize()
