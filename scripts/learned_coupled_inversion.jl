# author: Ziyi Yin, ziyi.yin@gatech.edu 

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

Random.seed!(3)
proj = false

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
x = mean(perm[:,:,1:ntrain], dims=3)[:,:,1];

# set up rock physics

vp = 3500 * ones(Float32,n)     # p-wave
phi = 0.25f0 * ones(Float32,n)  # porosity
rho = 2200 * ones(Float32,n)    # density
vp_stack = Patchy(sw_true,vp,rho,phi)[1]   # time-varying vp

##### Wave equation

d = (15f0, 15f0)        # discretization for wave equation
o = (0f0, 0f0)          # origin

extentx = (n[1]-1)*d[1] # width of model
extentz = (n[2]-1)*d[2] # depth of model

nsrc = 32       # num of sources
nrec = 960      # num of receivers

model = [Model(n, d, o, (1f3 ./ vp_stack[i]).^2f0; nb = 80) for i = 1:nv]   # wave model

timeS = timeR = 750f0               # recording time
dtS = dtR = 1f0                     # recording time sampling rate
ntS = Int(floor(timeS/dtS))+1       # time samples
ntR = Int(floor(timeR/dtR))+1       # source time samples

# source locations -- on the left hand side of the model
xsrc = convertToCell(range(1*d[1],stop=1*d[1],length=nsrc))
ysrc = convertToCell(range(0f0,stop=0f0,length=nsrc))
zsrc = convertToCell(range(d[2],stop=extentz,length=nsrc))

# receiver locations -- on the right hand side of the model
xrec = range(extentx,stop=extentx,length=nrec)
yrec = 0f0
zrec = range(d[2],stop=extentz,length=nrec)

# set up src/rec geometry
srcGeometry = Geometry(xsrc, ysrc, zsrc; dt=dtS, t=timeS)
recGeometry = Geometry(xrec, yrec, zrec; dt=dtR, t=timeR, nsrc=nsrc)

# set up source
f0 = 0.02f0     # kHz
wavelet = ricker_wavelet(timeS, dtS, f0)
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

# BackTracking linesearch algorithm
ls = BackTracking(c_1=1f-4,iterations=20,maxstep=Inf32,order=3,ρ_hi=5f-1,ρ_lo=1f-1)
fval = Inf32
α = 1f1

if proj
    # projection
    using SetIntersectionProjection
    ####### Projection TV + bound #######

    mutable struct compgrid
        d :: Tuple
        n :: Tuple
    end

    comp_grid = compgrid(d,n)

    options          = PARSDMM_options()
    options.FL       = Float32
    options.feas_tol = 0.001f0
    options.evol_rel_tol = 0.0001f0
    set_zero_subnormals(true)

    constraint = Vector{SetIntersectionProjection.set_definitions}()

    #bounds:
    m_min     = 10.0
    m_max     = 130.0
    set_type  = "bounds"
    TD_OP     = "identity"
    app_mode  = ("matrix","")
    custom_TD_OP = ([],false)
    push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP));

    # TV:
    TV = get_TD_operator(comp_grid,"TV",options.FL)[1]
    m_min     = 0.0
    m_max     = quantile([norm(TV*vec(perm[:,:,i]),1) for i = 1:ntrain], .8)
    set_type  = "l1"
    TD_OP     = "TV"
    app_mode  = ("matrix","")
    custom_TD_OP = ([],false)
    push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

    BLAS.set_num_threads(12)
    (P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid,options.FL)
    (TD_OP,AtA,l,y) = PARSDMM_precompute_distribute(TD_OP,set_Prop,comp_grid,options)

    function prj(x::Matrix{Float32})
        @time (x1,log_PARSDMM) = PARSDMM(vec(x),AtA,TD_OP,set_Prop,P_sub,comp_grid,options);
        return reshape(x1, n)::Matrix{Float32}
    end
else
    # just box constraints
    prj(x::Matrix{Float32}) = max.(min.(x,130f0),10f0)::Matrix{Float32}
end

### fluid-flow physics (FNO)
S(x::AbstractMatrix{Float32}) = permutedims(relu01(NN(perm_to_tensor(x, grid, AN)))[:,:,survey_indices,1], [3,1,2])

### rock physics
R(c::AbstractArray{Float32,3}) = Patchy(c,vp,rho,phi)[1]

### Define result directory
sim_name = "3D_FNO"
exp_name = "coupled_inversion"
plot_path = plotsdir(sim_name, exp_name)
save_path = datadir(sim_name, exp_name)

# iterations
grad_iterations = 100

# batchsize in wave equation, i.e. in each iteration the number of sources for each vintage to compute the gradient
nssample = 1

### track iterations
Loss = zeros(Float32,grad_iterations)
prog = Progress(grad_iterations)

for iter=1:grad_iterations

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
    function f(K)
        c = S(K); v = R(c); dpred = F(v);
        global loss = 0.5f0 * norm(dpred-dobs)^2f0
        return loss
    end

    ## AD by Flux
    @time g = gradient(()->f(x), Flux.params(x))

    # get the gradient from AD
    gvec = g.grads[x]::Matrix{Float32}
    p = -gvec/norm(gvec, Inf)

    # linesearch
    function ϕ(α)::Float32
        try
            global fval = f(prj(x + α * p))
        catch e
            @assert typeof(e) == DomainError
            global fval = Float32(Inf)
        end
        return fval
    end

    global α, fval = ls(ϕ, α, fval, dot(gvec, p))

    Loss[iter] = loss
    loss_iter = Loss[1:iter]

    # Update model and bound projection
    global x = prj(x + α * p)
    ProgressMeter.next!(prog; showvalues = [(:loss, fval), (:iter, iter), (:stepsize, α)])

    ### save intermediate results
    save_dict = @strdict iter x rand_ns α grad_iterations nv nsrc nrec proj loss_iter
    @tagsave(
        joinpath(save_path, savename(save_dict, "jld2"; digits=6)),
        save_dict;
        safe=true
    )

    ### plot current inverted permeability
    fig = figure(figsize=(20,12));
    subplot(1,3,1);
    plot_velocity(mean(perm[:,:,1:ntrain], dims=3)[:,:,1]', d; vmin=10f0, vmax=130f0, new_fig=false, name="initial");
    subplot(1,3,2);
    plot_velocity(x', d; vmin=10f0, vmax=130f0, new_fig=false, name="inversion after $iter iterations");
    subplot(1,3,3);
    plot_velocity(x_true', d; vmin=10f0, vmax=130f0, new_fig=false, name="ground truth");
    tight_layout()
    safesave(joinpath(plot_path, savename(save_dict; digits=6)*"_inversion.png"), fig);
    close(fig)

    ### save loss
    fig = figure(figsize=(20,12));
    plot(loss_iter);
    xlabel("iterations")
    ylabel("value")
    title("Objective function at iteration $iter")
    tight_layout();
    safesave(joinpath(plot_path, savename(save_dict; digits=6)*"_objective.png"), fig);
    close(fig)

end