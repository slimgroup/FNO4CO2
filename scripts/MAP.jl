# author: Ziyi Yin, ziyi.yin@gatech.edu 
## solving min ||F(G^{-1}(z))-y||_2^2 + λ^2 ||z||_2^2

using DrWatson
@quickactivate "FNO4CO2"

using FNO4CO2
using PyPlot
using Flux, Random
using MAT, Statistics, LinearAlgebra
using ProgressMeter, JLD2
using LineSearches
using InvertibleNetworks

Random.seed!(3)

# load the FNO network
JLD2.@load "../data/3D_FNO/batch_size=2_dt=0.02_ep=300_epochs=1000_learning_rate=0.0001_modes=4_nt=51_ntrain=1000_nvalid=100_s=1_width=20.jld2";
NN = deepcopy(NN_save);
Flux.testmode!(NN, true);

# load the NF network
JLD2.@load "../data/NFtrain/K=6_L=6_e=50_gab_l2=true_lr=0.001_lr_step=10_max_recursion=1_n_hidden=32_nepochs=500_noiseLev=0.02_λ=0.1.jld2";
G = NetworkMultiScaleHINT(1, n_hidden, L, K;
                               split_scales=true, max_recursion=max_recursion, p2=0, k2=1, activation=SigmoidLayer(low=0.5f0,high=1.0f0), logdet=false);
P_curr = get_params(G);
for j=1:length(P_curr)
    P_curr[j].data = Params[j].data;
end

# forward to set up splitting, take the reverse for Asim formulation
G(zeros(Float32,n[1],n[2],1,1));
G1 = reverse(G);

# Define raw data directory
mkpath(datadir("training-data"))
perm_path = datadir("training-data", "perm_gridspacing15.0.mat")
conc_path = datadir("training-data", "conc_gridspacing15.0.mat")

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

# physics grid
grid = gen_grid(n, d, nt, dt)

# take a test sample
x_true = perm[:,:,ntrain+nvalid+1];  # take a test sample
y_true = conc[:,:,:,ntrain+nvalid+1];

# observation vintages
nv = nt
survey_indices = Int.(round.(range(1, stop=nt, length=nv)))
yobs = permutedims(y_true[survey_indices,:,:,1:1],[2,3,1,4]); # ground truth CO2 concentration at these vintages

## add noise
noise_ = randn(Float32, size(yobs))
snr = 5f0
noise_ = noise_/norm(noise_) *  norm(yobs) * 10f0^(-snr/20f0)
σ = Float32.(norm(noise_)/sqrt(length(noise_)))
yobs = yobs + noise_

# initial z
x_init = 20f0 * ones(Float32, n);
x_init[:,25:36] .= 120f0;
z = vec(G(reshape(x_init, n[1], n[2], 1, 1)));
@time y_init = relu01(NN(perm_to_tensor(G1(z)[:,:,1,1], grid, AN)));

## weighting
λ = 1f0;

function S(x)
    return relu01(NN(perm_to_tensor(x, grid, AN)))[:,:,survey_indices,1];
end

# function value
function f(z)
    println("evaluate f")
    loss = .5f0/σ^2f0 * norm(S(G1(z)[:,:,1,1])-yobs)^2f0
    return loss
end

# set up plots
niterations = 50

hisloss = zeros(Float32, niterations+1)
hismisfit = zeros(Float32, niterations+1)
hisprior = zeros(Float32, niterations+1)
ls = BackTracking(c_1=1f-4,iterations=10,maxstep=Inf32,order=3,ρ_hi=5f-1,ρ_lo=1f-1)
α = 1f1;
### backtracking line search
prog = Progress(niterations)
for j=1:niterations

    p = Flux.params(z)

    @time grads = gradient(p) do
        global misfit = f(z)
        global prior = λ^2f0 * norm(z)^2f0/length(z)
        global loss = misfit + prior
        @show misfit, prior, loss
        println("evaluate g")
        return loss
    end
    if j == 1
        hisloss[1] = loss
        hismisfit[1] = misfit
        hisprior[1] = prior
    end
    g = grads.grads[z]
    gnorm = -g/norm(g, Inf)

    println("iteration no: ",j,"; function value: ",loss)

    # linesearch
    function ϕ(α)
        z1 = z .+ α .* gnorm
        global misfit = f(z1)
        global prior = λ^2f0 * norm(z1)^2f0/length(z1)
        global loss = misfit + prior
        @show misfit, prior, loss, α
        return loss
    end
    try
        global step, fval = ls(ϕ, α, loss, dot(g, gnorm))
    catch e
        println("linesearch failed at iteration: ",j)
        global niterations = j
        hisloss[j+1] = loss
        hismisfit[j+1] = misfit
        hisprior[j+1] = prior
        break
    end
    global α = 1.2f0 * step

    hisloss[j+1] = loss
    hismisfit[j+1] = misfit
    hisprior[j+1] = prior

    # Update model and bound projection
    global z .+= step .* gnorm

    ProgressMeter.next!(prog; showvalues = [(:loss, fval), (:misfit, misfit), (:prior, prior), (:iter, j), (:steplength, step)])

end

y_predict = S(G1(z)[:,:,1,1]);

## compute true and plot
SNR = -2f1 * log10(norm(x_true-G1(z)[:,:,1,1])/norm(x_true))
fig = figure(figsize=(20,12));
subplot(2,2,1);
imshow(G1(z)[:,:,1,1]',vmin=20,vmax=120);title("inversion by NN, $(niterations) iter");colorbar();
subplot(2,2,2);
imshow(x_true',vmin=20,vmax=120);title("GT permeability");colorbar();
subplot(2,2,3);
imshow(x_init',vmin=20,vmax=120);title("initial permeability");colorbar();
subplot(2,2,4);
imshow(5*abs.(x_true'-G1(z)[:,:,1,1]'),vmin=20,vmax=120);title("5X error, SNR=$SNR");colorbar();
suptitle("MAP (NF prior), snr=$snr")
tight_layout()

sim_name = "FNOinversion"
exp_name = "2phaseflow-NFprior"

save_dict = @strdict exp_name
plot_path = plotsdir(sim_name, savename(save_dict; digits=6))

fig_name = @strdict nv niterations λ α snr
safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_3Dfno_inv.png"), fig);

## loss
fig = figure(figsize=(20,12));
subplot(3,1,1);
plot(hisloss[1:niterations+1]);title("loss");
subplot(3,1,2);
plot(hismisfit[1:niterations+1]);title("misfit");
subplot(3,1,3);
plot(hisprior[1:niterations+1]);title("prior");
suptitle("MAP (NF prior), snr=$snr")
tight_layout()

safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_3Dfno_loss.png"), fig);

## data fitting
fig = figure(figsize=(20,12));
for i = 1:5
    subplot(4,5,i);
    imshow(y_init[:,:,10*i+1]', vmin=0, vmax=1);
    title("initial prediction at snapshot $(10*i+1)")
    subplot(4,5,i+5);
    imshow(yobs[:,:,10*i+1]', vmin=0, vmax=1);
    title("true at snapshot $(10*i+1)")
    subplot(4,5,i+10);
    imshow(y_predict[:,:,10*i+1]', vmin=0, vmax=1);
    title("predict at snapshot $(10*i+1)")
    subplot(4,5,i+15);
    imshow(5*abs.(yobs[:,:,10*i+1]'-y_predict[:,:,10*i+1]'), vmin=0, vmax=1);
    title("5X diff at snapshot $(10*i+1)")
end
suptitle("MAP (NF prior), snr=$snr")
tight_layout()
safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_3Dfno_fit.png"), fig);
