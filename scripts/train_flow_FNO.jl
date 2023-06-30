# author: Ziyi Yin, ziyi.yin@gatech.edu
# This script trains a Fourier Neural Operator which maps 2D permeability distribution to time-varying CO2 concentration snapshots.
# The PDE is in 2D while FNO requires 3D FFT

using DrWatson
@quickactivate "FNO4CO2"

using Pkg; Pkg.instantiate();

using FNO4CO2
using PyPlot
using JLD2
using Flux, Random, FFTW
using Statistics, LinearAlgebra
using CUDA
using ProgressMeter
using InvertibleNetworks:ActNorm
matplotlib.use("Agg")
using Printf: @printf

function meminfo_julia()
  @printf "GC total:  %9.3f MiB\n" Base.gc_total_bytes(Base.gc_num())/2^20
  @printf "GC live:   %9.3f MiB\n" Base.gc_live_bytes()/2^20
  @printf "JIT:       %9.3f MiB\n" Base.jit_total_bytes()/2^20
  @printf "Max. RSS:  %9.3f MiB\n" Sys.maxrss()/2^20
end
Random.seed!(1234)

# Define result directory

sim_name = "FNO-train"
exp_name = "flow-jutul"

plot_path = plotsdir(sim_name, exp_name)
slurm = try
    # Slurm
    parse(Int, ENV["SLURM_CPUS_ON_NODE"])
    true
catch e
    # Desktop
    false
end
save_path = slurm ? joinpath("/slimdata/zyin62/FNO-NF-MCMC/data/", sim_name, exp_name) : datadir(sim_name, exp_name)
mkpath(save_path)

# Define raw data directory
mkpath(datadir("gen-train","flow-channel"))
perm_path = joinpath(datadir("gen-train","flow-channel"), "irate=0.005_nsample=2000.jld2")

# Download the dataset into the data directory if it does not exist
if ~isfile(perm_path)
    run(`wget https://www.dropbox.com/s/8jb5g4rmamigoqf/'
        'irate=0.005_nsample=2000.jld2 -q -O $perm_path`)
end

dict_data = JLD2.jldopen(perm_path, "r")
perm = Float32.(dict_data["Ks"]);
conc = Float32.(dict_data["conc"]);

nsamples = size(perm, 3)

ntrain = 1900
nvalid = 100

batch_size = 4
learning_rate = 1f-4

epochs = 1000

modes = 4
width = 20

n = (64,64)
#d = (15f0,15f0) # dx, dy in m
d = (1f0/64, 1f0/64)

s = 1

nt = size(conc, 3)
#dt = 20f0    # dt in day
dt = 1f0/(nt-1)

AN = ActNorm(ntrain)
AN.forward(reshape(perm[1:s:end,1:s:end,1:ntrain], n[1], n[2], 1, ntrain));

y_train = conc[1:s:end,1:s:end,1:nt,1:ntrain];
y_valid = conc[1:s:end,1:s:end,1:nt,ntrain+1:ntrain+nvalid];

grid = gen_grid(n, d, nt, dt)

x_train = perm_to_tensor(perm[1:s:end,1:s:end,1:ntrain],grid,AN);
x_valid = perm_to_tensor(perm[1:s:end,1:s:end,ntrain+1:ntrain+nvalid],grid,AN);

# value, x, y, t

NN = Net3d(modes, width)
gpu_flag && (global NN = NN |> gpu);

Flux.trainmode!(NN, true);
w = Flux.params(NN)

opt = Flux.Optimise.ADAMW(learning_rate, (0.9f0, 0.999f0), 1f-4)
nbatches = Int(ntrain/batch_size)

Loss = zeros(Float32,epochs*nbatches)
Loss_valid = zeros(Float32, epochs)
prog = Progress(ntrain * epochs)

# plot figure
x_plot = x_valid[:, :, :, :, 1:1]
y_plot = y_valid[:, :, :, 1:1]

## training
λ = 1f0
for ep = 1:epochs
    meminfo_julia()
    Base.flush(Base.stdout)
    idx_e = reshape(randperm(ntrain), batch_size, nbatches)

    Flux.trainmode!(NN, true);
    for b = 1:nbatches
        x = x_train[:, :, :, :, idx_e[:,b]]
        y = y_train[:, :, :, idx_e[:,b]]
        if gpu_flag
            x = x |> gpu
            y = y |> gpu
        end
        grads = gradient(w) do
            ypred = relu01(NN(x))
            global loss = norm(ypred-y)/norm(y) + λ^2f0 * abs(sum(ypred)-sum(y))/sum(y)
            return loss
        end
        Loss[(ep-1)*nbatches+b] = loss
        for p in w
            Flux.Optimise.update!(opt, p, grads[p])
        end
        ProgressMeter.next!(prog; showvalues = [(:loss, loss), (:epoch, ep), (:batch, b)])
    end

    Flux.testmode!(NN, true);

    y_predict = relu01(NN(x_plot |> gpu))   |> cpu

    fig = figure(figsize=(20, 12))

    for i = 1:5
        subplot(4,5,i)
        imshow(x_plot[:,:,i+3,1,1]')
        title("x")

        subplot(4,5,i+5)
        imshow(y_plot[:,:,i+3,1]', vmin=0, vmax=1)
        title("true y")

        subplot(4,5,i+10)
        imshow(y_predict[:,:,i+3,1]', vmin=0, vmax=1)
        title("predict y")

        subplot(4,5,i+15)
        imshow(5f0 .* abs.(y_plot[:,:,i+3,1]'-y_predict[:,:,i+3,1]'), vmin=0, vmax=1)
        title("5X abs difference")

    end
    tight_layout()
    fig_name = @strdict ep batch_size Loss modes width learning_rate epochs s n d nt dt AN ntrain nvalid
    safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_3Dfno_fitting.png"), fig);
    close(fig)

    NN_save = NN |> cpu;
    w_save = Flux.params(NN_save)   

    valid_idx = randperm(nvalid)[1:batch_size]
    Loss_valid[ep] = norm(relu01(NN_save(x_valid[:, :, :, :, valid_idx])) - y_valid[:, :, :, valid_idx])/norm(y_valid[:, :, :, valid_idx])

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

    param_dict = @strdict ep NN_save w_save batch_size Loss modes width learning_rate epochs s n d nt dt AN ntrain nvalid loss_train loss_valid
    @tagsave(
        joinpath(save_path, savename(param_dict, "jld2"; digits=6)),
        param_dict;
        safe=true
    )

end
