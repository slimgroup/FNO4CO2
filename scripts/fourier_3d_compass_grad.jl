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
using LineSearches
using PyCall
@pyimport numpy.ma as ma
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

JLD2.@load "../data/3D_FNO/batch_size=20_computational_batch_size=2_dt=0.05_ep=150_epochs=500_learning_rate=0.0002_nt=21_ntrain=1000_nvalid=50_s=1_width=20.jld2";

y_train = permutedims(conc[1:nt,1:s:end,1:s:end,1:ntrain],[2,3,1,4]);
y_valid = permutedims(conc[1:nt,1:s:end,1:s:end,ntrain+1:ntrain+nvalid],[2,3,1,4]);

grid = gen_grid(n, d, nt, dt)

x_train = perm[1:s:end,1:s:end,1:ntrain];
x_valid = perm[1:s:end,1:s:end,ntrain+1:ntrain+nvalid];

qgrid_train = qgrid[:,1:ntrain];
qgrid_valid = qgrid[:,ntrain+1:ntrain+nvalid];

train_loader = Flux.Data.DataLoader((x_train, qgrid_train, y_train); batchsize = computational_batch_size, shuffle = true);
valid_loader = Flux.Data.DataLoader((x_valid, qgrid_valid, y_valid); batchsize = computational_batch_size, shuffle = true);

# load the network
JLD2.@load "../data/3D_FNO/batch_size=20_computational_batch_size=2_dt=0.05_ep=150_epochs=500_learning_rate=0.0002_nt=21_ntrain=1000_nvalid=50_s=1_width=20.jld2";
NN = deepcopy(NN_save);
Flux.testmode!(NN, true);
gpu_flag && (global NN = NN |> gpu);

opt = Flux.Optimise.ADAMW(learning_rate, (0.9f0, 0.999f0), 1f-4);
nbatches = Int(floor(ntrain/computational_batch_size))

Loss = zeros(Float32,epochs*nbatches)
Loss_valid = zeros(Float32, epochs)
prog = Progress(ntrain * epochs)

# Define result directory

sim_name = "FNOinversion"
exp_name = "2phaseflow-compass"

save_dict = @strdict exp_name
plot_path = plotsdir(sim_name, savename(save_dict; digits=6))

# plot figure
x_plot = x_valid[:, :, 1:1];
q_plot = qgrid_valid[:,1:1];
y_plot = y_valid[:, :, :, 1];

function q_tensorize(q::Matrix{Int64})
    q_tensor = zeros(Float32, n[1], n[2], nt, 1, size(q,2));
    for i = 1:size(q,2)
        q_tensor[q[1,i],q[2,i],:,1,i] .= 3f-1       ## q location, injection rate = 3f-1
    end
    return q_tensor |> gpu
end
q_tensorize(q::Vector{Int64}) = q_tensorize(reshape(q, :, 1))

@time y_predict = relu01(NN(cat(perm_to_tensor(x_plot|>gpu,grid,AN|>gpu), q_tensorize(q_plot), dims=4)))|>cpu;

function plotK(K, ax)
    p1 = pycall(ma.masked_greater, Any, K, 50)
    p2 = pycall(ma.masked_less, Any, K, 50)
    p3 = pycall(ma.masked_greater, Any, K, 10)

    ax.imshow(p2,interpolation="None",cmap="Reds",vmin=240,vmax=280)
    ax.imshow(p1,interpolation="None",cmap="winter",vmin=16,vmax=18);
    ax.imshow(p3,interpolation="None",cmap="Greys",vmin=0f0,vmax=1f-8)
end

fig, ax = subplots(4,5,figsize=(20, 12))

for i = 1:5
    ax[1, i][:axis]("off")
    plotK(x_plot[:,:,1]', ax[1,i])
    ax[1, i].set_title("x")

    ax[2, i][:axis]("off")
    ax[2, i].imshow(y_plot[:,:,4*i+1,1]', vmin=0, vmax=1)
    ax[2, i].set_title("true y")

    ax[3, i][:axis]("off")
    ax[3, i].imshow(y_predict[:,:,4*i+1,1]', vmin=0, vmax=1)
    ax[3, i].set_title("predict y")

    ax[4, i][:axis]("off")
    ax[4, i].imshow(5f0 .* abs.(y_plot[:,:,4*i+1,1]'-y_predict[:,:,4*i+1,1]'), vmin=0, vmax=1)
    ax[4, i].set_title("5X abs difference")

end

tight_layout()

# plot figure
x_true = perm[:, :, end];
q_true = qgrid[:,end];
y_true = permutedims(conc[:, :, :, end], [2,3,1]);
gpu_flag && (global y_true = y_true |> gpu);
x = mean(perm[:,:,1:ntrain], dims=3)[:,:,1] |> gpu;
gpu_flag && (global x = x |> gpu);
x_init = deepcopy(x) |> cpu;

qtensor = q_tensorize(q_true);
function S(x)
    return relu01(NN(cat(perm_to_tensor(x,grid,AN|>gpu), qtensor, dims=4)));
end

# function value
function f(x)
    println("evaluate f")
    loss = 0.5f0 * norm(S(x)-y_true)^2f0
    return loss
end

# set up plots
niterations = 100

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
        misfit = f(x .+ α .* gnorm)
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

y_predict = S(x);


sim_name = "FNOinversion"
exp_name = "2phaseflow-compass"

save_dict = @strdict exp_name
plot_path = plotsdir(sim_name, savename(save_dict; digits=6))

fig_name = @strdict niterations α
## compute true and plot
SNR = -2f1 * log10(norm(x_true-x)/norm(x_true))
fig, ax = subplots(2,2,figsize=(20, 12))

ax[1, 1][:axis]("off")
ax[1,1].imshow(x', vmin=0f0, vmax=300f0)
ax[1, 1].set_title("inversion by NN, $(niterations) iter")

ax[1, 2][:axis]("off")
ax[1,2].imshow(x_true', vmin=0f0, vmax=300f0)
ax[1, 2].set_title("GT permeability")

ax[2, 1][:axis]("off")
ax[2,1].imshow(x_init', vmin=0f0, vmax=300f0)
ax[2, 1].set_title("initial permeability")

ax[2, 2][:axis]("off")
ax[2,2].imshow(5*abs.(x'-x_init'), vmin=0f0, vmax=300f0)
ax[2, 2].set_title("5X Update")

suptitle("MLE (no prior)")
tight_layout()
safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_3Dfno_update.png"), fig);

## compute true and plot
SNR = -2f1 * log10(norm(x_true-x)/norm(x_true))
fig, ax = subplots(2,2,figsize=(20, 12))

ax[1, 1][:axis]("off")
plotK(x', ax[1,1])
ax[1, 1].set_title("inversion by NN, $(niterations) iter")

ax[1, 2][:axis]("off")
plotK(x_true[:,:,1]', ax[1,2])
ax[1, 2].set_title("GT permeability")

ax[2, 1][:axis]("off")
plotK(x_init', ax[2,1])
ax[2, 1].set_title("initial permeability")

ax[2, 2][:axis]("off")
plotK(5*abs.(x_true'-x'), ax[2,2])
ax[2, 2].set_title("5X error, SNR=$SNR")

suptitle("MLE (no prior)")
tight_layout()

safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_3Dfno_inv.png"), fig);


## loss
fig = figure(figsize=(20,12));
plot(hisloss[1:niterations+1]);title("loss");
suptitle("MLE (no prior)")
tight_layout()

safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_3Dfno_loss.png"), fig);

y_init = S(x_init);
## data fitting
fig = figure(figsize=(20,12));
for i = 1:5
    subplot(4,5,i);
    imshow(y_init[:,:,4*i+1]', vmin=0, vmax=1);
    title("initial prediction at snapshot $(4*i+1)")
    subplot(4,5,i+5);
    imshow(y_true[:,:,4*i+1]', vmin=0, vmax=1);
    title("true at snapshot $(4*i+1)")
    subplot(4,5,i+10);
    imshow(y_predict[:,:,4*i+1]', vmin=0, vmax=1);
    title("predict at snapshot $(4*i+1)")
    subplot(4,5,i+15);
    imshow(5*abs.(y_true[:,:,4*i+1]'-y_predict[:,:,4*i+1]'), vmin=0, vmax=1);
    title("5X diff at snapshot $(4*i+1)")
end
suptitle("MLE (no prior)")
tight_layout()

safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_3Dfno_fit.png"), fig);


