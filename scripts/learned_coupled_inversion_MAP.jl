# author: Ziyi Yin, ziyi.yin@gatech.edu 
## This script conducts learned coupled inversion where time-lapse seismic data is used to directly invert for the permeability in the Earth subsurface
## Three units are involved in the learned coupled inversion framework, namely
## - a pre-trained FNO as a surrogate for the fluid flow simulator, which maps permeability to time evolution of CO2 concentration
## - a rock physics model that maps each CO2 concentration to acoustic velocity of the rock
## - a wave physics model that generates acoustic seismic data from the velocity (using JUDI)
## In particular, this script uses a pre-trained normalizing flow as a prior for permeability models
## and uses Asim's formulation to invert for the latent variable -- check http://proceedings.mlr.press/v119/asim20a/asim20a.pdf for more details

using DrWatson
@quickactivate "FNO4CO2"

using FNO4CO2
using PyPlot
using Flux, Random
using MAT, Statistics, LinearAlgebra
using ProgressMeter, JLD2
using LineSearches
using InvertibleNetworks:ActNorm
using JUDI
using SlimPlotting
using InvertibleNetworks

Random.seed!(2022)
matplotlib.use("agg")

# load the network
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
nv = 11
survey_indices = Int.(round.(range(1, stop=51, length=nv)))
sw_true = y_true[survey_indices,:,:]; # ground truth CO2 concentration at these vintages

# initial z
z = zeros(Float32, prod(n));
x_init = G1(z)[:,:,1,1];

# set up rock physics

vp = 3500 * ones(Float32,n)     # p-wave
phi = 0.25f0 * ones(Float32,n)  # porosity
rho = 2200 * ones(Float32,n)    # density
vp_stack = Patchy(sw_true,vp,rho,phi)[1]   # time-varying vp

## upsampling
upsample = 5
u(x::Vector{Matrix{Float32}}) = [repeat(x[i], inner=(upsample,upsample)) for i = 1:nv]
vp_stack_up = u(vp_stack)

##### Wave equation
n = n.*upsample
d = (15f0/upsample, 15f0/upsample)        # discretization for wave equation
o = (0f0, 0f0)          # origin

extentx = (n[1]-1)*d[1] # width of model
extentz = (n[2]-1)*d[2] # depth of model

nsrc = 32       # num of sources
nrec = 960      # num of receivers

model = [Model(n, d, o, (1f3 ./ vp_stack_up[i]).^2f0; nb = 160) for i = 1:nv]   # wave model

timeS = timeR = 750f0               # recording time
dtS = dtR = 1f0                     # recording time sampling rate
ntS = Int(floor(timeS/dtS))+1       # time samples
ntR = Int(floor(timeR/dtR))+1       # source time samples

mode = "both"
if mode == "reflection"
    xsrc = convertToCell(range(d[1],stop=(n[1]-1)*d[1],length=nsrc))
    zsrc = convertToCell(range(10f0,stop=10f0,length=nsrc))
    xrec = range(d[1],stop=(n[1]-1)*d[1],length=nrec)
    zrec = range(10f0,stop=10f0,length=nrec)
elseif mode == "transmission"
    xsrc = convertToCell(range(d[1],stop=d[1],length=nsrc))
    zsrc = convertToCell(range(d[2],stop=(n[2]-1)*d[2],length=nsrc))
    xrec = range((n[1]-1)*d[1],stop=(n[1]-1)*d[1], length=nrec)
    zrec = range(d[2],stop=(n[2]-1)*d[2],length=nrec)
else
    # source locations -- half at the left hand side of the model, half on top
    xsrc = convertToCell(vcat(range(d[1],stop=d[1],length=Int(nsrc/2)),range(d[1],stop=(n[1]-1)*d[1],length=Int(nsrc/2))))
    zsrc = convertToCell(vcat(range(d[2],stop=(n[2]-1)*d[2],length=Int(nsrc/2)),range(10f0,stop=10f0,length=Int(nsrc/2))))
    xrec = vcat(range((n[1]-1)*d[1],stop=(n[1]-1)*d[1], length=Int(nrec/2)),range(d[1],stop=(n[1]-1)*d[1],length=Int(nrec/2)))
    zrec = vcat(range(d[2],stop=(n[2]-1)*d[2],length=Int(nrec/2)),range(10f0,stop=10f0,length=Int(nrec/2)))
end

ysrc = convertToCell(range(0f0,stop=0f0,length=nsrc))
yrec = 0f0

# set up src/rec geometry
srcGeometry = Geometry(xsrc, ysrc, zsrc; dt=dtS, t=timeS)
recGeometry = Geometry(xrec, yrec, zrec; dt=dtR, t=timeR, nsrc=nsrc)

# set up source
f0 = 0.05f0     # kHz
wavelet = ricker_wavelet(timeS, dtS, f0)
q = judiVector(srcGeometry, wavelet)

# set up simulation operators
Ftrue = [judiModeling(model[i], srcGeometry, recGeometry) for i = 1:nv] # acoustic wave equation solver

# Define seismic data directory
mkpath(datadir("seismic-data"))
misc_dict = @strdict nsrc nrec upsample

### generate/load data
if ~isfile(datadir("seismic-data", savename(misc_dict, "jld2"; digits=6)))
    println("generating data")
    global d_obs = [Ftrue[i]*q for i = 1:nv]
    seismic_dict = @strdict nsrc nrec upsample d_obs q srcGeometry recGeometry model
    @tagsave(
        datadir("seismic-data", savename(seismic_dict, "jld2"; digits=6)),
        seismic_dict;
        safe=true
    )
else
    println("loading data")
    JLD2.@load datadir("seismic-data", savename(misc_dict, "jld2"; digits=6)) d_obs
    global d_obs = d_obs
end

## add noise
noise_ = deepcopy(d_obs)
for i = 1:nv
    for j = 1:nsrc
        noise_[i].data[j] = randn(Float32, ntR, nrec)
    end
end
snr = 1000f0
noise_ = noise_/norm(noise_) *  norm(d_obs) * 10f0^(-snr/20f0)
σ = 1f0
d_obs = d_obs + noise_

### fluid-flow physics (FNO)
S(x::AbstractMatrix{Float32}) = permutedims(relu01(NN(perm_to_tensor(x, grid, AN)))[:,:,survey_indices,1], [3,1,2])

### rock physics
R(c::AbstractArray{Float32,3}) = Patchy(c,vp,rho,phi)[1]

### init concentration and seismic
@time y_init = S(x_init);
rand_ns = [[1] for i = 1:nv]                             # select random source idx for each vintage
q_sub = [q[rand_ns[i]] for i = 1:nv]                                        # set-up source
F_sub = [Ftrue[i][rand_ns[i]] for i = 1:nv]                                 # set-up wave modeling operator

### wave physics
function F_init(v::Vector{Matrix{Float32}})
    m = [vec(1f3./v[i]).^2f0 for i = 1:nv]
    return [F_sub[i](m[i], q_sub[i]) for i = 1:nv]
end
v_init = R(y_init); v_up_init = u(v_init); d_init = F_init(v_up_init);

### Define result directory
sim_name = "coupled_inversion"
exp_name = "NFprior"
plot_path = plotsdir(sim_name, exp_name)
save_path = datadir(sim_name, exp_name)

# iterations
niterations = 100

# batchsize in wave equation, i.e. in each iteration the number of sources for each vintage to compute the gradient
nssample = 8

### track iterations
hisloss = zeros(Float32, niterations+1)
hismisfit = zeros(Float32, niterations+1)
hisprior = zeros(Float32, niterations+1)
prog = Progress(niterations)

## weighting
λ = 0f0;

θ = Flux.params(z)

# ADAM-W algorithm
learning_rate = 2f-2
lr_step   = 10
lr_rate = 0.75f0
opt = Flux.Optimiser(ExpDecay(learning_rate, lr_rate, nsrc/nssample*lr_step, 1f-6), ADAMW(learning_rate))

for iter=1:niterations

    Base.flush(Base.stdout)
    rand_ns = [jitter(nsrc, nssample) for i = 1:nv]                             # select random source idx for each vintage
    q_sub = [q[rand_ns[i]] for i = 1:nv]                                        # set-up source
    F_sub = [Ftrue[i][rand_ns[i]] for i = 1:nv]                                 # set-up wave modeling operator
    dobs = [d_obs[i][rand_ns[i]] for i = 1:nv]                                  # subsampled seismic dataset from the selected sources

    ### wave physics
    function F(v::Vector{Matrix{Float32}})
        m = [vec(1f3./v[i]).^2f0 for i = 1:nv]
        return [F_sub[i](m[i], q_sub[i]) for i = 1:nv]
    end

    # function value
    function f(z)
        x = G1(z)[:,:,1,1]; c = S(x); v = R(c); v_up = u(v); dpred = F(v_up);
        global misfit = .5f0/σ^2f0 * nsrc/nssample * norm(dpred-dobs)^2f0
        global prior = λ^2f0 * norm(z)^2f0/length(z)
        global fval = misfit + prior
        @show misfit, prior, fval
        return fval
    end

    ## AD by Flux
    @time grads = gradient(()->f(z), θ)
    for p in θ
        Flux.Optimise.update!(opt, p, grads[p])
    end

    hisloss[iter] = fval
    hismisfit[iter] = misfit
    hisprior[iter] = prior

    y_predict = S(G1(z)[:,:,1,1]);
    rand_ns = [[1] for i = 1:nv]                             # select random source idx for each vintage
    q_sub = [q[rand_ns[i]] for i = 1:nv]                                        # set-up source
    F_sub = [Ftrue[i][rand_ns[i]] for i = 1:nv]                                 # set-up wave modeling operator
    dobs = [d_obs[i][rand_ns[i]] for i = 1:nv]                                  # subsampled seismic dataset from the selected sources

    ### wave physics
    function F(v::Vector{Matrix{Float32}})
        m = [vec(1f3./v[i]).^2f0 for i = 1:nv]
        return [F_sub[i](m[i], q_sub[i]) for i = 1:nv]
    end
    v = R(y_predict); v_up = u(v); dpred = F(v_up);

    ProgressMeter.next!(prog; showvalues = [(:loss, fval), (:misfit, misfit), (:prior, prior), (:iter, iter), (:stepsize, step)])

    ### save intermediate results
    save_dict = @strdict iter snr nssample z λ rand_ns step niterations nv nsrc nrec survey_indices hisloss hismisfit hisprior learning_rate lr_step lr_rate
    @tagsave(
        joinpath(save_path, savename(save_dict, "jld2"; digits=6)),
        save_dict;
        safe=true
    )

    ## save figure
    fig_name = @strdict iter snr nssample λ niterations nv nsrc nrec survey_indices learning_rate lr_step lr_rate

    ## compute true and plot
    SNR = -2f1 * log10(norm(x_true-G1(z)[:,:,1,1])/norm(x_true))
    fig = figure(figsize=(20,12));
    subplot(2,2,1);
    imshow(G1(z)[:,:,1,1]',vmin=20,vmax=120);title("inversion by NN, $(iter) iter");colorbar();
    subplot(2,2,2);
    imshow(x_true',vmin=20,vmax=120);title("GT permeability");colorbar();
    subplot(2,2,3);
    imshow(x_init',vmin=20,vmax=120);title("initial permeability");colorbar();
    subplot(2,2,4);
    imshow(5*abs.(x_true'-G1(z)[:,:,1,1]'),vmin=20,vmax=120);title("5X error, SNR=$SNR");colorbar();
    suptitle("Learned Coupled Inversion MAP (NF prior) at iter $iter, seismic data snr=$snr")
    tight_layout()
    safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_inv.png"), fig);
    close(fig)

    ## loss
    fig = figure(figsize=(20,12));
    subplot(3,1,1);
    plot(hisloss[1:iter]);title("loss=$(hisloss[iter])");
    subplot(3,1,2);
    plot(hismisfit[1:iter]);title("misfit=$(hismisfit[iter])");
    subplot(3,1,3);
    plot(hisprior[1:iter]);title("prior=$(hisprior[iter])");
    suptitle("Learned Coupled Inversion MAP (NF prior) at iter $iter, seismic data snr=$snr")
    tight_layout()
    safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_loss.png"), fig);
    close(fig)

    ## CO2 fitting
    fig = figure(figsize=(20,12));
    for i = 1:5
        subplot(4,5,i);
        imshow(y_init[i,:,:]', vmin=0, vmax=1);
        title("initial prediction at snapshot $(survey_indices[i])")
        subplot(4,5,i+5);
        imshow(sw_true[i,:,:]', vmin=0, vmax=1);
        title("true at snapshot $(survey_indices[i])")
        subplot(4,5,i+10);
        imshow(y_predict[i,:,:]', vmin=0, vmax=1);
        title("predict at snapshot $(survey_indices[i])")
        subplot(4,5,i+15);
        imshow(5*abs.(sw_true[i,:,:]'-y_predict[i,:,:]'), vmin=0, vmax=1);
        title("5X diff at snapshot $(survey_indices[i])")
    end
    suptitle("Learned Coupled Inversion MAP (NF prior) at iter $iter, seismic data snr=$snr")
    tight_layout()
    safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_3Dfno_fit.png"), fig);
    close(fig)

    ## seismic data fitting
    fig = figure(figsize=(20,12));
    for i = 1:5
        subplot(4,5,i);
        plot_sdata(d_init[i].data[1], (1f0, 1f0); new_fig=false);colorbar();
        title("initial prediction at snapshot $(survey_indices[i])")
        subplot(4,5,i+5);
        plot_sdata(dobs[i].data[1], (1f0, 1f0); new_fig=false);colorbar();
        title("true at snapshot $(survey_indices[i])")
        subplot(4,5,i+10);
        plot_sdata(dpred[i].data[1], (1f0, 1f0); new_fig=false);colorbar();
        title("predict at snapshot $(survey_indices[i])")
        subplot(4,5,i+15);
        plot_sdata(dobs[i].data[1]-dpred[i].data[1], (1f0, 1f0); new_fig=false);colorbar();
        title("diff at snapshot $(survey_indices[i])")
    end
    suptitle("Learned Coupled Inversion MAP (NF prior) at iter $iter, seismic data snr=$snr")
    tight_layout()
    safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_3Dfno_data_fit.png"), fig);
    close(fig)

    GC.gc()

end
