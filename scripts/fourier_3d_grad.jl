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
using SetIntersectionProjection

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
nv = nt
survey_indices = Int.(round.(range(1, stop=nt, length=nv)))
yobs = permutedims(y_true[survey_indices,:,:,1:1],[2,3,1,4]); # ground truth CO2 concentration at these vintages

# initial x
x = 20f0 * ones(Float32, n);
x[:,25:36] .= 120f0;
x_init = deepcopy(x);
@time y_init = relu01(NN(perm_to_tensor(x, grid, AN)));

function S(x)
    return relu01(NN(perm_to_tensor(x, grid, AN)));
end

# function value
function f(x)
    println("evaluate f")
    loss = 0.5f0 * norm(S(x)-yobs)^2f0
    return loss
end

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

# set up plots
niterations = 50

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
        x1 = prj(x .+ α .* gnorm)
        misfit = f(x1)
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
    global x .= prj(x .+ step .* gnorm)

    ProgressMeter.next!(prog; showvalues = [(:loss, fval), (:iter, j), (:steplength, step)])

end

y_predict = S(x);

## compute true and plot
SNR = -2f1 * log10(norm(x_true-x)/norm(x_true))
fig = figure(figsize=(20,12));
subplot(2,2,1);
imshow(x',vmin=20,vmax=120);title("inversion by NN, $(niterations) iter");colorbar();
subplot(2,2,2);
imshow(x_true',vmin=20,vmax=120);title("GT permeability");colorbar();
subplot(2,2,3);
imshow(x_init',vmin=20,vmax=120);title("initial permeability");colorbar();
subplot(2,2,4);
imshow(5*abs.(x_true'-x'),vmin=20,vmax=120);title("5X error, SNR=$SNR");colorbar();
suptitle("MLE (no prior)")
tight_layout()

sim_name = "FNOinversion"
exp_name = "2phaseflow"

save_dict = @strdict exp_name
plot_path = plotsdir(sim_name, savename(save_dict; digits=6))

fig_name = @strdict proj nv niterations α
safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_3Dfno_inv.png"), fig);


## loss
fig = figure(figsize=(20,12));
plot(hisloss[1:niterations+1]);title("loss");
suptitle("MLE (no prior)")
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
suptitle("MLE (no prior)")
tight_layout()
safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_3Dfno_fit.png"), fig);


