# author: Ziyi Yin, ziyi.yin@gatech.edu 
## This script conducts learned coupled inversion where time-lapse seismic data is used to directly invert for the permeability in the Earth subsurface
## Three units are involved in the learned coupled inversion framework, namely
## - a pre-trained FNO as a surrogate for the fluid flow simulator, which maps permeability to time evolution of CO2 concentration
## - a rock physics model that maps each CO2 concentration to acoustic velocity of the rock
## - a wave physics model that generates acoustic seismic data from the velocity (using JUDI)

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

Random.seed!(2022)
proj = false
matplotlib.use("agg")

# load the network
JLD2.@load "../data/3D_FNO/batch_size=20_computational_batch_size=2_dt=0.05_ep=150_epochs=500_learning_rate=0.0002_nt=21_ntrain=1000_nvalid=50_s=1_width=20.jld2";
NN = deepcopy(NN_save);
Flux.testmode!(NN, true);

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

y_train = permutedims(conc[1:nt,1:s:end,1:s:end,1:ntrain],[2,3,1,4]);
y_valid = permutedims(conc[1:nt,1:s:end,1:s:end,ntrain+1:ntrain+nvalid],[2,3,1,4]);

x_train = perm[1:s:end,1:s:end,1:ntrain];
x_valid = perm[1:s:end,1:s:end,ntrain+1:ntrain+nvalid];

qgrid_train = qgrid[:,1:ntrain];
qgrid_valid = qgrid[:,ntrain+1:ntrain+nvalid];

# physics grid
grid = gen_grid(n, d, nt, dt)

# Define result directory

sim_name = "coupled-inversion"
exp_name = "2phaseflow-compass"

save_dict = @strdict exp_name
plot_path = plotsdir(sim_name, savename(save_dict; digits=6))

function q_tensorize(q::Matrix{Int64})
    q_tensor = zeros(Float32, n[1], n[2], nt, 1, size(q,2));
    for i = 1:size(q,2)
        q_tensor[q[1,i],q[2,i],:,1,i] .= 3f-1       ## q location, injection rate = 3f-1
    end
    return q_tensor
end
q_tensorize(q::Vector{Int64}) = q_tensorize(reshape(q, :, 1))

# take a test sample
x_true = perm[:,:,ntrain+nvalid+1];  # take a test sample
y_true = conc[:,:,:,ntrain+nvalid+1];

# observation vintages
nv = 11
survey_indices = Int.(round.(range(2, stop=18, length=nv)))
sw_true = y_true[survey_indices,:,:]; # ground truth CO2 concentration at these vintages

# initial x
x_init = mean(perm[:,:,1:ntrain],dims=3)[:,:,1];
q_true = qgrid[:,ntrain+nvalid+1:ntrain+nvalid+1];
qtensor = q_tensorize(q_true);
x = deepcopy(x_init);
@time y_init = relu01(NN(cat(perm_to_tensor(x_init,grid,AN), qtensor, dims=4)));

# set up rock physics

JLD2.@load datadir("coupled-inversion-compass","v_test.jld2") v_test
vp = v_test
JLD2.@load datadir("coupled-inversion-compass","rho_test.jld2") rho_test
rho = rho_test * 1f3

phi = 0.25f0 * ones(Float32,n)  # porosity

vp_stack = Patchy(sw_true,vp,rho,phi; bulk_min = 5f10)[1]   # time-varying vp

##### Wave equation
d = (12f0, 12f0)        # discretization for wave equation
o = (0f0, 0f0)          # origin

extentx = (n[1]-1)*d[1] # width of model
extentz = (n[2]-1)*d[2] # depth of model

nsrc = 64       # num of sources
nrec = 336      # num of receivers

model = [Model(n, d, o, (1f3 ./ vp_stack[i]).^2f0; nb = 120) for i = 1:nv]   # wave model

timeS = timeR = 3600f0               # recording time
dtS = dtR = 4f0                     # recording time sampling rate
ntS = Int(floor(timeS/dtS))+1       # time samples
ntR = Int(floor(timeR/dtR))+1       # source time samples


mode = "reflection"
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
f0 = 0.015f0     # kHz
wavelet = ricker_wavelet(timeS, dtS, f0)    # need low-filter later
q = judiVector(srcGeometry, wavelet)

# set up simulation operators
Ftrue = [judiModeling(model[i], srcGeometry, recGeometry) for i = 1:nv] # acoustic wave equation solver

# Define seismic data directory
mkpath(datadir("seismic-data"))
misc_dict = @strdict nsrc nrec

### generate/load data
if ~isfile(datadir("seismic-data", savename(misc_dict, "jld2"; digits=6)))
    println("generating data")
    global d_obs = [Ftrue[i]*q for i = 1:nv]
    seismic_dict = @strdict nsrc nrec d_obs q srcGeometry recGeometry model
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

# BackTracking linesearch algorithm
ls = BackTracking(c_1=1f-4,iterations=10,maxstep=Inf32,order=3,ρ_hi=5f-1,ρ_lo=1f-1)
fval = Inf32
α = 1f1

### fluid-flow physics (FNO)
S(x::AbstractMatrix{Float32}) = permutedims(relu01(NN(cat(perm_to_tensor(x, grid, AN), qtensor, dims=4)))[:,:,survey_indices,1], [3,1,2])

### rock physics
R(c::AbstractArray{Float32,3}) = Patchy(c,vp,rho,phi; bulk_min = 5f10)[1]

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
v_init = R(y_init); d_init = F_init(v_init);

### Define result directory
sim_name = "coupled-inversion-compass"
exp_name = "no_prior"
plot_path = plotsdir(sim_name, exp_name)
save_path = datadir(sim_name, exp_name)

# just box constraints
prj(x::Matrix{Float32}) = max.(min.(x,302f0),0f0)::Matrix{Float32}

# iterations
niterations = 200

# batchsize in wave equation, i.e. in each iteration the number of sources for each vintage to compute the gradient
nssample = 8

### track iterations
hisloss = zeros(Float32, niterations+1)
prog = Progress(niterations)

# ADAM-W algorithm
learning_rate = 5f-3
lr_step   = 10
lr_rate = 0.75f0
opt = Flux.Optimiser(ExpDecay(learning_rate, lr_rate, nsrc/nssample*lr_step, 1f-6), ADAMW(learning_rate))

for iter=1:niterations

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
    function f(x)
        c = S(x); v = R(c); dpred = F(v);
        global fval = .5f0/σ^2f0 * nsrc/nssample * norm(dpred-dobs)^2f0
        @show fval
        return fval
    end

    ## AD by Flux
    θ = Flux.params(x)
    @time g = gradient(()->f(x), θ)
    for p in θ
        Flux.Optimise.update!(opt, p, g[p])
    end
    global x = prj(x)
    
    ## initial loss
    if iter == 1
        hisloss[1] = fval
    end
    hisloss[iter+1] = fval

    # Update model and bound projection

    y_predict = S(x);
    rand_ns = [[1] for i = 1:nv]                             # select random source idx for each vintage
    q_sub = [q[rand_ns[i]] for i = 1:nv]                                        # set-up source
    F_sub = [Ftrue[i][rand_ns[i]] for i = 1:nv]                                 # set-up wave modeling operator
    dobs = [d_obs[i][rand_ns[i]] for i = 1:nv]                                  # subsampled seismic dataset from the selected sources

    ### wave physics
    function F(v::Vector{Matrix{Float32}})
        m = [vec(1f3./v[i]).^2f0 for i = 1:nv]
        return [F_sub[i](m[i], q_sub[i]) for i = 1:nv]
    end
    v_predict = R(y_predict); dpred = F(v_predict);

    ProgressMeter.next!(prog; showvalues = [(:loss, fval), (:iter, iter), (:stepsize, step)])

    ### save intermediate results
    save_dict = @strdict mode iter snr nssample x rand_ns step niterations nv nsrc nrec survey_indices hisloss learning_rate lr_step lr_rate
    @tagsave(
        joinpath(save_path, savename(save_dict, "jld2"; digits=6)),
        save_dict;
        safe=true
    )

    ## save figure
    fig_name = @strdict mode proj iter snr nssample niterations nv nsrc nrec survey_indices

    ## compute true and plot
    SNR = -2f1 * log10(norm(x_true-x)/norm(x_true))
    fig = figure(figsize=(20,12));
    subplot(2,2,1);
    imshow(x',vmin=20,vmax=120);title("inversion by NN, SNR=$SNR");colorbar();
    subplot(2,2,2);
    imshow(x_true',vmin=20,vmax=120);title("GT permeability");colorbar();
    subplot(2,2,3);
    imshow(x_init',vmin=20,vmax=120);title("initial permeability");colorbar();
    subplot(2,2,4);
    imshow(5*(x'-x_init'),vmin=20,vmax=120);title("5X inversion - initial");colorbar();
    suptitle("Learned Coupled Inversion at iter $iter, seismic data snr=$snr")
    tight_layout()
    safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_inv.png"), fig);
    close(fig)

    ## loss
    fig = figure(figsize=(20,12));
    plot(hisloss[1:iter+1]);title("loss=$(hisloss[iter+1])");
    suptitle("Learned Coupled Inversion at iter $iter, seismic data snr=$snr")
    tight_layout()
    safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_loss.png"), fig);
    close(fig)

    ## CO2 fitting
    fig = figure(figsize=(20,12));
    for i = 1:5
        subplot(4,5,i);
        imshow(y_init[2*i,:,:]', vmin=0, vmax=1);
        title("initial prediction at snapshot $(survey_indices[2*i])")
        subplot(4,5,i+5);
        imshow(sw_true[2*i,:,:]', vmin=0, vmax=1);
        title("true at snapshot $(survey_indices[2*i])")
        subplot(4,5,i+10);
        imshow(y_predict[2*i,:,:]', vmin=0, vmax=1);
        title("predict at snapshot $(survey_indices[2*i])")
        subplot(4,5,i+15);
        imshow(5*abs.(sw_true[2*i,:,:]'-y_predict[2*i,:,:]'), vmin=0, vmax=1);
        title("5X diff at snapshot $(survey_indices[2*i])")
    end
    suptitle("Learned Coupled Inversion at iter $iter, seismic data snr=$snr")
    tight_layout()
    safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_3Dfno_fit.png"), fig);
    close(fig)

    ## seismic data fitting
    fig = figure(figsize=(20,12));
    for i = 1:5
        subplot(4,5,i);
        plot_sdata(d_init[2*i].data[1], (1f0, 1f0); new_fig=false);colorbar();
        title("initial prediction at snapshot $(survey_indices[2*i])")
        subplot(4,5,i+5);
        plot_sdata(dobs[2*i].data[1], (1f0, 1f0); new_fig=false);colorbar();
        title("true at snapshot $(survey_indices[2*i])")
        subplot(4,5,i+10);
        plot_sdata(dpred[2*i].data[1], (1f0, 1f0); new_fig=false);colorbar();
        title("predict at snapshot $(survey_indices[2*i])")
        subplot(4,5,i+15);
        plot_sdata(dobs[2*i].data[1]-dpred[2*i].data[1], (1f0, 1f0); new_fig=false);colorbar();
        title("diff at snapshot $(survey_indices[2*i])")
    end
    suptitle("Learned Coupled Inversion MAP (NF prior) at iter $iter, seismic data snr=$snr")
    tight_layout()
    safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_3Dfno_data_fit.png"), fig);
    close(fig)

    GC.gc()

end
