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
using Seis4CCS.RockPhysics
using JUDI
using SlimPlotting

Random.seed!(2022)
matplotlib.use("agg")

# load the network
JLD2.@load "../data/3D_FNO/batch_size=1_dt=0.02_ep=200_epochs=200_learning_rate=0.0001_modes=4_nt=51_ntrain=1000_nvalid=100_s=1_width=20.jld2";
NN = deepcopy(NN_save);
Flux.testmode!(NN, true);

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
survey_indices = Int.(round.(range(1, stop=22, length=nv)))
sw_true = y_true[survey_indices,:,:]; # ground truth CO2 concentration at these vintages

# initial x
x_init = 20f0 * ones(Float32, n);
x_init[:,25:36] .= 120f0;
x = deepcopy(x_init);
@time y_init = relu01(NN(perm_to_tensor(x, grid, AN)));

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

model = [Model(n, d, o, (1f3 ./ vp_stack_up[i]).^2f0; nb = 80) for i = 1:nv]   # wave model

timeS = timeR = 750f0               # recording time
dtS = dtR = 1f0                     # recording time sampling rate
ntS = Int(floor(timeS/dtS))+1       # time samples
ntR = Int(floor(timeR/dtR))+1       # source time samples

# source locations -- half at the left hand side of the model, half on top
xsrc = convertToCell(vcat(range(d[1],stop=d[1],length=Int(nsrc/2)),range(d[1],stop=(n[1]-1)*d[1],length=Int(nsrc/2))))
ysrc = convertToCell(range(0f0,stop=0f0,length=nsrc))
zsrc = convertToCell(vcat(range(d[2],stop=(n[2]-1)*d[2],length=Int(nsrc/2)),range(10f0,stop=10f0,length=Int(nsrc/2))))

# receiver locations -- half at the right hand side of the model, half on top
xrec = vcat(range((n[1]-1)*d[1],stop=(n[1]-1)*d[1], length=Int(nrec/2)),range(d[1],stop=(n[1]-1)*d[1],length=Int(nrec/2)))
yrec = 0f0
zrec = vcat(range(d[2],stop=(n[2]-1)*d[2],length=Int(nrec/2)),range(10f0,stop=10f0,length=Int(nrec/2)))

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
snr = 10f0
noise_ = noise_/norm(noise_) *  norm(d_obs) * 10f0^(-snr/20f0)
σ = Float32.(norm(noise_)/sqrt(length(noise_)))
d_obs = d_obs + noise_

# BackTracking linesearch algorithm
ls = BackTracking(c_1=1f-4,iterations=10,maxstep=Inf32,order=3,ρ_hi=5f-1,ρ_lo=1f-1)
fval = Inf32
α = 1f1

### fluid-flow physics (FNO)
S(x::AbstractMatrix{Float32}) = permutedims(relu01(NN(perm_to_tensor(x, grid, AN)))[:,:,survey_indices,1], [3,1,2])

### rock physics
R(c::AbstractArray{Float32,3}) = Patchy(c,vp,rho,phi)[1]

### Define result directory
sim_name = "coupled_inversion"
exp_name = "no_prior"
plot_path = plotsdir(sim_name, exp_name)
save_path = datadir(sim_name, exp_name)

# iterations
niterations = 100

# batchsize in wave equation, i.e. in each iteration the number of sources for each vintage to compute the gradient
nssample = 8

### track iterations
hisloss = zeros(Float32, niterations+1)
prog = Progress(niterations)

## initial steplength
α = 1f1;

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
        c = S(x); v = R(c); v_up = u(v); dpred = F(v_up);
        global fval = .5f0/σ^2f0 * nsrc/nssample * norm(dpred-dobs)^2f0
        @show fval
        return fval
    end

    ## AD by Flux
    @time g = gradient(()->f(x), Flux.params(x)).grads[x]
    
    ## initial loss
    if iter == 1
        hisloss[1] = fval
    end

    # (normalized) update direction
    p = -g/norm(g, Inf)

    # linesearch
    function ϕ(α)::Float32
        try
            global fval = f(x + α * p)
        catch e
            @assert typeof(e) == DomainError
            global fval = Inf32
        end
        return fval
    end

    try
        global step, fval = ls(ϕ, α, fval, dot(g, p))
    catch e
        println("linesearch failed at iteration: ",j)
        global niterations = j
        hisloss[j+1] = fval
        break
    end

    global α = 1.2f0 * step

    hisloss[iter+1] = fval

    # Update model and bound projection
    global x .+= step .* p

    y_predict = S(x);

    ProgressMeter.next!(prog; showvalues = [(:loss, fval), (:iter, iter), (:stepsize, step)])

    ### save intermediate results
    save_dict = @strdict iter snr nssample x rand_ns step niterations nv nsrc nrec survey_indices hisloss
    @tagsave(
        joinpath(save_path, savename(save_dict, "jld2"; digits=6)),
        save_dict;
        safe=true
    )

    ## save figure
    fig_name = @strdict iter snr nssample niterations nv nsrc nrec survey_indices

    ## compute true and plot
    SNR = -2f1 * log10(norm(x_true-x)/norm(x_true))
    fig = figure(figsize=(20,12));
    subplot(2,2,1);
    imshow(x',vmin=20,vmax=120);title("inversion by NN, $(iter) iter");colorbar();
    subplot(2,2,2);
    imshow(x_true',vmin=20,vmax=120);title("GT permeability");colorbar();
    subplot(2,2,3);
    imshow(x_init',vmin=20,vmax=120);title("initial permeability");colorbar();
    subplot(2,2,4);
    imshow(5*abs.(x_true'-x'),vmin=20,vmax=120);title("5X error, SNR=$SNR");colorbar();
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

    ## data fitting
    fig = figure(figsize=(20,12));
    for i = 1:5
        subplot(4,5,i);
        imshow(y_init[:,:,survey_indices[2*i],1]', vmin=0, vmax=1);
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

end
