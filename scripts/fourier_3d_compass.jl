# author: Ziyi Yin, ziyi.yin@gatech.edu
# This script trains a Fourier Neural Operator which maps 2D permeability distribution to time-varying CO2 concentration snapshots.
# The PDE is in 2D while FNO requires 3D FFT

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
matplotlib.use("Agg")

Random.seed!(1234)

# Define raw data directory
mkpath(datadir("training-data"))
perm_path = datadir("training-data", "perm_compass.jld2")
conc_path = datadir("training-data", "conc_compass.jld2")
qgrid_path = datadir("training-data", "qgrid_compass.jld2")

# Download the dataset into the data directory if it does not exist
if ~isfile(perm_path)
    run(`wget https://www.dropbox.com/s/71r9oidb10georq/'
        'perm_compass.jld2 -q -O $perm_path`)
end
if ~isfile(conc_path)
    run(`wget https://www.dropbox.com/s/kuuuyjhj439cynq/'
        'conc_compass.jld2 -q -O $conc_path`)
end
if ~isfile(qgrid_path)
    run(`wget https://www.dropbox.com/s/9sekkojxt8fyslo/'
        'qgrid_compass.jld2 -q -O $qgrid_path`)
end

JLD2.@load perm_path perm;
JLD2.@load conc_path conc;
JLD2.@load qgrid_path qgrid;

nsamples = size(perm, 3)

ntrain = 1000
nvalid = 50

batch_size = 20         # effective one
computational_batch_size = 2 # of samples that still fit on GPU
grad_accum_iter = batch_size/computational_batch_size   # accumulate these many gradients
learning_rate = 2f-4

epochs = 500

modes = [32, 16, 8]
width = 20

n = (size(perm, 1), size(perm, 2))
#d = (12f0,12f0) # dx, dy in m
d = 1f0./n

s = 1

nt = size(conc,1)
#dt = 20f0    # dt in day
dt = 1f0/(nt-1)

AN = ActNorm(ntrain)
norm_perm = AN.forward(reshape(perm[1:s:end,1:s:end,1:ntrain], n[1], n[2], 1, ntrain));

y_train = permutedims(conc[1:nt,1:s:end,1:s:end,1:ntrain],[2,3,1,4]);
y_valid = permutedims(conc[1:nt,1:s:end,1:s:end,ntrain+1:ntrain+nvalid],[2,3,1,4]);

grid = gen_grid(n, d, nt, dt)

x_train = perm[1:s:end,1:s:end,1:ntrain];
x_valid = perm[1:s:end,1:s:end,ntrain+1:ntrain+nvalid];

qgrid_train = qgrid[:,1:ntrain];
qgrid_valid = qgrid[:,ntrain+1:ntrain+nvalid];

train_loader = Flux.Data.DataLoader((x_train, qgrid_train, y_train); batchsize = computational_batch_size, shuffle = true)
valid_loader = Flux.Data.DataLoader((x_valid, qgrid_valid, y_valid); batchsize = computational_batch_size, shuffle = true)

NN = Net3d(modes, width; in_channels=5)
gpu_flag && (global NN = NN |> gpu)

Flux.trainmode!(NN, true)
w = Flux.params(NN)

opt = Flux.Optimise.ADAMW(learning_rate, (0.9f0, 0.999f0), 1f-4)
nbatches = Int(floor(ntrain/computational_batch_size))

Loss = zeros(Float32,epochs*nbatches)
Loss_valid = zeros(Float32, epochs)
prog = Progress(ntrain * epochs)

# Define result directory

sim_name = "3D_FNO"
exp_name = "2phaseflow-compass"

save_dict = @strdict exp_name
plot_path = plotsdir(sim_name, savename(save_dict; digits=6))

# plot figure
x_plot = x_valid[:, :, 1:1]
q_plot = qgrid_valid[:,1:1]
y_plot = y_valid[:, :, :, 1]

function q_tensorize(q::Matrix{Int64})
    q_tensor = zeros(Float32, n[1], n[2], nt, 1, size(q,2));
    for i = 1:size(q,2)
        q_tensor[q[1,i],q[2,i],:,1,i] .= 3f-1       ## q location, injection rate = f-1
    end
    return q_tensor
end

## training
iter = 0
for ep = 1:epochs
    Base.flush(Base.stdout)

    ## update
    Flux.trainmode!(NN, true);
    for (x,q,y) in train_loader
        global iter = iter + 1
        x = cat(perm_to_tensor(x,grid,AN), q_tensorize(q), dims=4)
        if gpu_flag
            x = x |> gpu
            y = y |> gpu
        end
        grads = gradient(w) do
            global loss = norm(relu01(NN(x))-y)/norm(y)
            return loss
        end
        (iter==1) && (global grads_sum = 0f0 .* grads)
        global grads_sum = grads_sum .+ grads
        Loss[iter] = loss
        if mod(iter, grad_accum_iter) == 0
            for p in w
                Flux.Optimise.update!(opt, p, grads_sum[p])
            end
            grads_sum = 0f0 .* grads_sum
        end
        ProgressMeter.next!(prog; showvalues = [(:loss, loss), (:epoch, ep), (:iter, iter)])
    end

    Flux.testmode!(NN, true);
    y_predict = relu01(NN(cat(perm_to_tensor(x_plot,grid,AN), q_tensorize(q), dims=4)|>gpu))|>cpu

    fig, ax = subplots(4,5,figsize=(20, 12))

    for i = 1:5
        ax[1, i][:axis]("off")
        ax[1, i].imshow(x_plot[:,:,1]')
        title("x")

        ax[2, i][:axis]("off")
        ax[2, i].imshow(y_plot[:,:,4*i+1,1]', vmin=0, vmax=1)
        title("true y")

        ax[3, i][:axis]("off")
        ax[3, i].imshow(y_predict[:,:,4*i+1,1]', vmin=0, vmax=1)
        title("predict y")

        ax[4, i][:axis]("off")
        ax[4, i].imshow(5f0 .* abs.(y_plot[:,:,4*i+1,1]'-y_predict[:,:,4*i+1,1]'), vmin=0, vmax=1)
        title("5X abs difference")

    end
    tight_layout()
    fig_name = @strdict ep batch_size Loss modes width learning_rate epochs s n d nt dt AN ntrain nvalid computational_batch_size
    safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_3Dfno_fitting.png"), fig);
    close(fig)

    NN_save = NN |> cpu
    w_save = Flux.params(NN_save)   

    for (x,q,y) in valid_loader
        Loss_valid[ep] = norm(relu01(NN_save(cat(perm_to_tensor(x,grid,AN), q_tensorize(q), dims=4)))-y)/norm(y)
        break
    end

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

    param_dict = @strdict ep NN_save w_save batch_size Loss modes width learning_rate epochs s n d nt dt AN ntrain nvalid loss_train loss_valid computational_batch_size
    @tagsave(
        datadir(sim_name, savename(param_dict, "jld2"; digits=6)),
        param_dict;
        safe=true
    )
    
end

NN_save = NN |> cpu;
w_save = params(NN_save);

final_dict = @strdict Loss Loss_valid epochs NN_save w_save batch_size Loss modes width learning_rate s n d nt dt AN ntrain nvalid computational_batch_size;

@tagsave(
    datadir(sim_name, savename(final_dict, "jld2"; digits=6)),
    final_dict;
    safe=true
);
