# author: Ziyi (Francis) Yin
# This script trains a Fourier Neural Operator which maps 2D permeability distribution to time-varying CO2 concentration snapshots.
# The PDE is in 2D while FNO requires 3D FFT

using DrWatson
@quickactivate "FNO"
import Pkg; Pkg.instantiate()

using PyPlot
using Flux, Random, FFTW, Zygote, NNlib
using MAT, Statistics, LinearAlgebra
using CUDA
using ProgressMeter
using InvertibleNetworks:ActNorm
matplotlib.use("Agg")

try
    CUDA.device()
    global gpu_flag=true
catch e
    println("CUDA.device() found no GPU device on this machine.")
    global gpu_flag=false
end

gpu_flag = false

include("utils.jl")
include("fno3dstruct.jl")

Random.seed!(1234)

ntrain = 1000
nvalid = 100

batch_size = 10
learning_rate = 1f-4

epochs = 200
step_size = 100
gamma = 5f-1

modes = 4
width = 20

n = (64,64)
#d = (15f0,15f0) # dx, dy in m
d = (1f0/64, 1f0/64)

s = 1

nt = 51
#dt = 20f0    # dt in day
dt = 1f0/(nt-1)

# Define raw data directory
mkpath(datadir("data"))
perm_path = datadir("data", "perm_gridspacing15.0.mat")
conc_path = datadir("data", "conc_gridspacing15.0.mat")

# Download the dataset into the data directory if it does not exist
if ~isfile(perm_path)
    run(`wget https://www.dropbox.com/s/eqre95eqggqkdq2/'
        'perm_gridspacing15.0.mat -q -O $perm_path`)
end
if ~isfile(conc_path)
    run(`wget https://www.dropbox.com/s/b5zkp6cw60bd4lt/'
        'conc_gridspacing15.0.mat -q -O $conc_path`)
end

perm = matread(perm_path)["perm"];
conc = matread(conc_path)["conc"];

AN = ActNorm(ntrain)

x_train_ = AN.forward(reshape(perm[1:s:end,1:s:end,1:ntrain], n[1], n[2], 1, ntrain))
x_valid_ = AN.forward(reshape(perm[1:s:end,1:s:end,ntrain+1:ntrain+nvalid], n[1], n[2], 1, nvalid))

y_train = permutedims(conc[1:nt,1:s:end,1:s:end,1:ntrain],[2,3,1,4]);
y_valid = permutedims(conc[1:nt,1:s:end,1:s:end,ntrain+1:ntrain+nvalid],[2,3,1,4]);

grid = zeros(Float32,n[1],n[2],2)
grid[:,:,1] = repeat(reshape(collect(range(d[1],stop=n[1]*d[1],length=n[1])), :, 1)',n[2])' # x
grid[:,:,2] = repeat(reshape(collect(range(d[2],stop=n[2]*d[2],length=n[2])), 1, :),n[1])   # z

x_train = zeros(Float32,n[1],n[2],nt,4,ntrain);
x_valid = zeros(Float32,n[1],n[2],nt,4,nvalid);

for i = 1:nt
    x_train[:,:,i,1,:] = deepcopy(x_train_)
    x_valid[:,:,i,1,:] = deepcopy(x_valid_)
    for j = 1:ntrain
        x_train[:,:,i,2,j] = grid[:,:,1]
        x_train[:,:,i,3,j] = grid[:,:,2]
        x_train[:,:,i,4,j] .= (i-1)*dt
    end

    for k = 1:nvalid
        x_valid[:,:,i,2,k] = grid[:,:,1]
        x_valid[:,:,i,3,k] = grid[:,:,2]
        x_valid[:,:,i,4,k] .= (i-1)*dt
    end
end

# value, x, y, t

train_loader = Flux.Data.DataLoader((x_train, y_train); batchsize = batch_size, shuffle = true)
valid_loader = Flux.Data.DataLoader((x_valid, y_valid); batchsize = batch_size, shuffle = true)

NN = Net3d(modes, width)
gpu_flag && (global NN = NN |> gpu)

Flux.trainmode!(NN, true)
w = Flux.params(NN)

opt = Flux.Optimise.ADAMW(learning_rate, (0.9f0, 0.999f0), 1f-4)
nbatches = Int(ntrain/batch_size)

Loss = zeros(Float32,epochs*nbatches)
Loss_valid = zeros(Float32, epochs)
prog = Progress(ntrain * epochs)

# plot figure
x_plot = x_valid[:, :, :, :, 1:1]
y_plot = y_valid[:, :, :, 1:1]

# Define result directory

sim_name = "3D_FNO"
exp_name = "2phaseflow"

save_dict = @strdict exp_name
plot_path = plotsdir(sim_name, savename(save_dict; digits=6))

## training

iter = 0
for ep = 1:epochs

    Base.flush(Base.stdout)
    idx_e = reshape(randperm(ntrain), batch_size, nbatches)

    for b = 1:nbatches
        x = x_train[:, :, :, :, idx_e[:,b]]
        y = y_train[:, :, :, idx_e[:,b]]
        if gpu_flag
            x = x |> gpu
            y = y |> gpu
        end
        global iter = iter + 1
        grads = gradient(w) do
            global loss = Flux.mse(relu01(NN(x)),y)
            return loss
        end
        Loss[iter] = loss
        for p in w
            Flux.Optimise.update!(opt, p, grads[p])
        end
        ProgressMeter.next!(prog; showvalues = [(:loss, loss), (:epoch, ep), (:batch, b)])
    end

    valid_idx = randperm(nvalid)[1:batch_size]
    x_v = x_valid[:, :, :, :, valid_idx]
    y_v = y_valid[:, :, :, valid_idx]
    y_predict = relu01(NN(x_v))
    Loss_valid[ep] = Flux.mse(y_predict, y_v)

    y_predict = relu01(NN(x_plot))
    fig = figure(figsize=(20, 12))

    for i = 1:5
        subplot(4,5,i)
        imshow(x_plot[:,:,10*i+1,1,1]')
        title("x")

        subplot(4,5,i+5)
        imshow(y_plot[:,:,10*i+1,1]', vmin=0, vmax=1)
        title("true y")

        subplot(4,5,i+10)
        imshow(y_predict[:,:,10*i+1,1]', vmin=0, vmax=1)
        title("predict y")

        subplot(4,5,i+10)
        imshow(5f0 .* abs.(y_plot[:,:,10*i+1,1]'-y_predict[:,:,10*i+1,1]'), vmin=0, vmax=1)
        title("5X abs difference")

    end
    tight_layout()
    fig_name = @strdict ep NN w batch_size Loss modes width learning_rate epochs gamma step_size s n d nt dt AN
    safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_3Dfno_fitting.png"), fig);
    close(fig)

    fig = figure(figsize=(20, 12))
    plot(Loss[1:nbatches*ep]);
    plot(1:nbatches:nbatches*ep, Loss_valid[1:ep]); 
    title("Objective function at epoch " + ep)
    legend(["training", "validation"])
    tight_layout();
    safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_3Dfno_loss.png"), fig);
    close(fig);

    NN_save = NN |> cpu
    w_save = convert.(Array,w |> cpu)
    param_dict = @strdict NN_save w_save batch_size Loss modes width learning_rate epochs gamma step_size s n d nt dt AN
    @tagsave(
        datadir(sim_name, savename(param_dict, "jld2"; digits=6)),
        param_dict;
        safe=true
    )
end
