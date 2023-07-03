using DrWatson
@quickactivate "FNO4CO2"
using Pkg; Pkg.instantiate()
using InvertibleNetworks
using JLD2
using Random
using Flux: gpu, cpu
using Flux.Data: DataLoader
using Flux.Optimise: Optimiser, ClipNorm, ExpDecay, update!, ADAM
using LinearAlgebra
using CUDA
using PyPlot
using Distributions

matplotlib.use("agg")
slurm = try
    # Slurm
    parse(Int, ENV["SLURM_CPUS_ON_NODE"])
    true
catch e
    # Desktop
    false
end

sim_name = "NF-train"
exp_name = "perm-channel"

save_path = slurm ? joinpath("/slimdata/zyin62/FNO4CO2/data/", sim_name, exp_name) : datadir(sim_name, exp_name)
mkpath(save_path)

# Define raw data directory
perm_path = joinpath(save_path, "cons=1e-5_delta=25_num_sample=10000_theta0=5.jld2")

# Download the dataset into the data directory if it does not exist
if ~isfile(perm_path)
    run(`wget https://www.dropbox.com/s/xy36bvoz6iqau60/'
        'cons=1e-5_delta=25_num_sample=10000_theta0=5.jld2 -q -O $perm_path`)
end

dict_data = JLD2.jldopen(perm_path, "r")
perm = dict_data["perm"]

function z_shape_simple(G, ZX_test)
    Z_save, ZX = split_states(ZX_test[:], G.Z_dims)
    for i=G.L:-1:1
        if i < G.L
            ZX = tensor_cat(ZX, Z_save[i])
        end
        ZX = G.squeezer.inverse(ZX) 
    end
    ZX
end

# Training hyperparameters
nepochs    = 512
batch_size = 50
lr        = 1f-3
lr_step   = 10
gab_l2 = true
λ = 1f-1

#architecture parametrs
L = 5
K = 6
n_hidden   = 32
low       = 0.5f0
max_recursion = 1
clip_norm = 5f0

#data augmentation
α = 0.005f0
β = 0.5f0
αmin = 0.005f0

# Random seed
Random.seed!(2022)

# load in training dataz
ntrain = 2000
nvalid = Int(round(0.05 * size(perm, 3)))

train_x = perm[:,:,1:ntrain];
X_train   = reshape(Float32.(train_x), size(train_x)[1], size(train_x)[2], 1, size(train_x)[3]);

nx, ny, nc, _ = size(X_train);
test_x = perm[:,:,ntrain+1:ntrain+nvalid];
X_test   = reshape(Float32.(test_x), size(test_x)[1], size(test_x)[2], 1, size(test_x)[3]);
N = nx*ny 

# Create network
G = NetworkMultiScaleHINT(1, n_hidden, L, K;
                               split_scales=true, max_recursion=max_recursion, p2=0, k2=1, activation=SigmoidLayer(low=low,high=1.0f0))|> gpu

# Test latent 
X_train_latent = X_train[:,:,:,1:batch_size];
X_test_latent  = X_test[:,:,:,1:batch_size];

X_train_latent = X_train_latent |> gpu;
X_test_latent  = X_test_latent  |> gpu;

# Test generative samples 
ZX_noise = randn(Float32, nx, ny, nc, batch_size) |> gpu;

# Split in training/testing
#use all as training set because there is a separate testing set
train_idx = randperm(ntrain)[1:ntrain]

# Training
# Batch extractor
nbatches = cld(ntrain, batch_size)
train_loader = DataLoader(train_idx, batchsize=batch_size, shuffle=false)

# Optimizer
opt = Optimiser(ExpDecay(lr, .99f0, nbatches*lr_step, 1f-6), ClipNorm(clip_norm),ADAM(lr))

t = G.forward(X_train_latent); # to initialize actnorm
θ = get_params(G);
θ_backup = deepcopy(θ);

# Training log keeper
floss = zeros(Float32, nbatches, nepochs);
fbdim = zeros(Float32, nbatches, nepochs);
flogdet = zeros(Float32, nbatches, nepochs);

floss_test = zeros(Float32, nepochs);
flogdet_test = zeros(Float32, nepochs);

intermediate_save_params = 5

for e=1:nepochs
    # Epoch-adaptive regularization weight
    λ_adaptive = λ*2*N/norm(θ_backup)^2

    idx_e = reshape(randperm(ntrain), batch_size, nbatches)
    for b = 1:nbatches # batch loop
        Base.flush(Base.stdout)

        X = X_train[:, :, :, idx_e[:,b]]

        X = X |> gpu

        noiseLev = α/((e-1)*nbatches+b)^β+αmin

        X .+= noiseLev*CUDA.randn(Float32, size(X))*norm(X_train, Inf)

        Zx, lgdet = G.forward(X)

        floss[b,e]   = norm(Zx)^2 / (N*batch_size)
        flogdet[b,e] = lgdet / (-N)

        G.backward((Zx / batch_size)[:], (Zx)[:])
        GC.gc()

        print("Iter: epoch=", e, "/", nepochs, ", batch=", b, "/", nbatches, 
            "; f l2 = ",  floss[b,e], 
            "; lgdet = ", flogdet[b,e], "; f = ", floss[b,e] + flogdet[b,e], "\n")

        for i =1:length(θ)
            Δθ = θ[i].data-θ_backup[i].data
            update!(opt, θ[i].data, θ[i].grad+λ_adaptive*Δθ)
            (b == nbatches) && (θ_backup[i].data .= θ[i].data)
        end

        clear_grad!(G)
    end

    ############# Test Image generation

    # Evaluate network on test dataset
    ZX_test, lgdet_test = G.forward(X_test_latent ) |> cpu;
    ZX_test_sq = z_shape_simple(G, ZX_test);

    flogdet_test[e] = lgdet_test / (-N)
    floss_test[e] = norm(ZX_test)^2f0 / (N*batch_size);

    # Evaluate network on train dataset
    ZX_train = G.forward(X_train_latent)[1] |> cpu;
    ZX_train_sq = z_shape_simple(G, ZX_train);

    #### make figures of generative samples
    X_gen = G.inverse(ZX_noise[:]) |> cpu;

    # Plot latent vars and qq plots. 
    mean_train_1 = round(mean(ZX_train_sq[:,:,1,1]),digits=2)
    std_train_1 = round(std(ZX_train_sq[:,:,1,1]),digits=2)

    mean_test_1 = round(mean(ZX_test_sq[:,:,1,1]),digits=2)
    std_test_1 = round(std(ZX_test_sq[:,:,1,1]),digits=2)

    mean_train_2 = round(mean(ZX_train_sq[:,:,1,2]),digits=2)
    std_train_2 = round(std(ZX_train_sq[:,:,1,2]),digits=2)

    mean_test_2 = round(mean(ZX_test_sq[:,:,1,2]),digits=2)
    std_test_2 = round(std(ZX_test_sq[:,:,1,2]),digits=2)


    fig = figure(figsize=(14, 12))

    subplot(4,5,1); imshow(X_gen[:,:,1,1]', aspect=1, vmin=20,vmax=120,resample=true, interpolation="none", filterrad=1, cmap="gray")
    axis("off");  title(L"$x\sim p_{\theta}(x)$")

    subplot(4,5,2); imshow(X_gen[:,:,1,2]', aspect=1, vmin=20,vmax=120,resample=true, interpolation="none", filterrad=1, cmap="gray")
    axis("off"); title(L"$x\sim p_{\theta}(x)$")

    subplot(4,5,3); imshow(X_train_latent[:,:,1,1]' |> cpu, aspect=1, vmin=20,vmax=120, resample=true, interpolation="none", filterrad=1, cmap="gray")
    axis("off"); title(L"$x_{train1} \sim p(x)$")

    subplot(4,5,4); imshow(ZX_train_sq[:,:,1,1]', aspect=1, resample=true, interpolation="none", filterrad=1, 
                                    vmin=-3, vmax=3, cmap="seismic"); axis("off"); 
    title(L"$z_{train1} = G^{-1}(x_{train1})$ "*string("\n")*" mean "*string(mean_train_1)*" std "*string(std_train_1));

    obs = vec(ZX_test_sq[:, :, 1, 1])
    F⁰ = Normal(0,1)
    nobs=length(obs); sort!(obs)
    quantiles⁰ = [quantile(F⁰,i/nobs) for i in 1:nobs]

    subplot(4,5,5);
    PyPlot.scatter(quantiles⁰, obs, s=0.5)
    PyPlot.plot(obs,obs,color="red",label="")
    title(L"qq plot with $z_{train1}$"); xlim(-5,5); ylim(-5,5);
    #xlabel("Theoretical Quantiles"); ylabel("Sample Quantiles")
    
    subplot(4,5,6); imshow(X_gen[:,:,1,3]', aspect=1, vmin=20,vmax=120,resample=true, interpolation="none", filterrad=1, cmap="gray")
    axis("off");  title(L"$x\sim p_{\theta}(x)$")

    subplot(4,5,7); imshow(X_gen[:,:,1,4]', aspect=1, vmin=20,vmax=120,resample=true, interpolation="none", filterrad=1, cmap="gray")
    axis("off"); title(L"$x\sim p_{\theta}(x)$")

    subplot(4,5,8); imshow(X_train_latent[:,:,1,2]' |> cpu, aspect=1, vmin=20,vmax=120, resample=true, interpolation="none", filterrad=1, cmap="gray")
    axis("off"); title(L"$x_{train2} \sim p(x)$")

      
    subplot(4,5,9) ;imshow(ZX_train_sq[:,:,1,2]', aspect=1, resample=true, interpolation="none", filterrad=1, 
                                    vmin=-3, vmax=3, cmap="seismic"); axis("off"); 
    title(L"$z_{train2} = G^{-1}(x_{train2})$ "*string("\n")*" mean "*string(mean_train_2)*" std "*string(std_train_2));
        
    obs = vec(ZX_test_sq[:, :, 1, 2])
    F⁰ = Normal(0,1)
    nobs=length(obs); sort!(obs)
    quantiles⁰ = [quantile(F⁰,i/nobs) for i in 1:nobs]

    subplot(4,5,10); PyPlot.scatter(quantiles⁰, obs, s=0.5)
    PyPlot.plot(obs,obs,color="red",label="")
    title(L"qq plot with $z_{train2}$"); xlim(-5,5);ylim(-5,5);
    #xlabel("Theoretical Quantiles"); ylabel("Sample Quantiles")  

    subplot(4,5,11); imshow(X_gen[:,:,1,5]', aspect=1, vmin=20,vmax=120,resample=true, interpolation="none", filterrad=1, cmap="gray")
    axis("off");  title(L"$x\sim p_{\theta}(x)$")

    subplot(4,5,12); imshow(X_gen[:,:,1,6]', aspect=1, vmin=20,vmax=120,resample=true, interpolation="none", filterrad=1, cmap="gray")
    axis("off"); title(L"$x\sim p_{\theta}(x)$")

    subplot(4,5,13); imshow(X_test_latent[:,:,1,1]' |> cpu, aspect=1, vmin=20,vmax=120, resample=true, interpolation="none", filterrad=1, cmap="gray")
    axis("off"); title(L"$x_{test1} \sim p(x)$")

      
    subplot(4,5,14) ;imshow(ZX_test_sq[:,:,1,1]', aspect=1, resample=true, interpolation="none", filterrad=1, 
                                    vmin=-3, vmax=3, cmap="seismic"); axis("off"); 
    title(L"$z_{test1} = G^{-1}(x_{test1})$ "*string("\n")*" mean "*string(mean_test_1)*" std "*string(std_test_1));
        
    obs = vec(ZX_test_sq[:, :, 1, 1])
    F⁰ = Normal(0,1)
    nobs=length(obs); sort!(obs)
    quantiles⁰ = [quantile(F⁰,i/nobs) for i in 1:nobs]

    subplot(4,5,15); PyPlot.scatter(quantiles⁰, obs, s=0.5)
    PyPlot.plot(obs,obs,color="red",label="")
    title(L"qq plot with $z_{test1}$"); xlim(-5,5);ylim(-5,5);
    #xlabel("Theoretical Quantiles"); ylabel("Sample Quantiles")  


    subplot(4,5,16); imshow(X_gen[:,:,1,7]', aspect=1, vmin=20,vmax=120,
        interpolation="none", filterrad=1, cmap="gray"); axis("off");  title(L"$x\sim p_{\theta}(x)$")

    subplot(4,5,17); imshow(X_gen[:,:,1,8]', aspect=1, vmin=20,vmax=120, interpolation="none", filterrad=1, cmap="gray")
    axis("off"); title(L"$x\sim p_{\theta}(x)$")

    subplot(4,5,18); imshow(X_test_latent[:,:,1,2]' |> cpu, aspect=1, vmin=20,vmax=120, interpolation="none", filterrad=1, cmap="gray")
    axis("off"); title(L"$x_{test2} \sim p(x)$")
     
    subplot(4,5,19) ;imshow(ZX_test_sq[:,:,1,2]', aspect=1, interpolation="none", filterrad=1, 
                                    vmin=-3, vmax=3, cmap="seismic"); axis("off"); 
    title(L"$z_{test2} = G^{-1}(x_{test2})$ "*string("\n")*" mean "*string(mean_test_2)*" std "*string(std_test_2));
        
    obs = vec(ZX_test_sq[:, :, 1, 2])
    F⁰ = Normal(0,1)
    nobs=length(obs); sort!(obs)
    quantiles⁰ = [quantile(F⁰,i/nobs) for i in 1:nobs]

    subplot(4,5,20); PyPlot.scatter(quantiles⁰, obs, s=0.5)
    PyPlot.plot(obs,obs,color="red",label="")
    title(L"qq plot with $z_{test2}$"); xlim(-5,5);ylim(-5,5);
    #xlabel("Theoretical Quantiles"); ylabel("Sample Quantiles")  

    tight_layout()

    fig_name = @strdict ntrain nvalid e gab_l2 λ lr lr_step α αmin β n_hidden L K max_recursion clip_norm
    safesave(joinpath(plotsdir(sim_name, exp_name), savename(fig_name; digits=6)*"_hint_latent.png"), fig); close(fig)
    close(fig)


    #save params every 20 epochs
    if(mod(e,intermediate_save_params)==0) 
         # Saving parameters and logs
         Params = get_params(G) |> cpu 
         save_dict = @strdict ntrain nvalid e nepochs lr lr_step gab_l2 λ α αmin β n_hidden L K max_recursion Params floss flogdet clip_norm
         @tagsave(
             joinpath(save_path, savename(save_dict, "jld2"; digits=6)),
             save_dict;
             safe=true
         )
    end

    ############# Training metric logs
    vfloss   =  vec(floss)
    vflogdet =  vec(flogdet)
    vsum = vfloss + vflogdet

    vfloss_test   =  vec(floss_test)
    vflogdet_test =  vec(flogdet_test)
    vsum_test = vfloss_test + vflogdet_test

    fig = figure("training logs ", figsize=(10,8))
    vfloss_epoch = vfloss[1:findall(x -> x == 0,vfloss)[1]-1]
    vfloss_epoch_test = vfloss_test[1:findall(x -> x == 0,vfloss_test)[1]-1]

    subplot(3,1,1)
    title("L2 Term: train="*string(vfloss_epoch[end])*" test="*string(vfloss_epoch_test[end]))
    plot(vfloss_epoch, label="train");
    plot(1:nbatches:nbatches*e, vfloss_epoch_test, label="test"); 
    axhline(y=1f0,color="red",linestyle="--",label="Noise Likelihood")
    ylim(0.5,1.5); xlabel("Parameter Update"); legend()
    
    subplot(3,1,2)
    vflogdet_epoch = vflogdet[1:findall(x -> x == 0,vflogdet)[1]-1]
    vflogdet_epoch_test = vflogdet_test[1:findall(x -> x == 0,vflogdet_test)[1]-1]
    title("Logdet Term: train="*string(vflogdet_epoch[end])*" test="*string(vflogdet_epoch_test[end]))
    plot(vflogdet_epoch);
    plot(1:nbatches:nbatches*e, vflogdet_epoch_test);
    xlabel("Parameter Update")  

    subplot(3,1,3)
    vsum_epoch = vsum[1:findall(x -> x == 0,vsum)[1]-1]
    vsum_epoch_test = vsum_test[1:findall(x -> x == 0,vsum_test)[1]-1]
    plot(vsum_epoch); title("Total Objective: train="*string(vsum_epoch[end])*" test="*string(vsum_epoch_test[end]))
    plot(1:nbatches:nbatches*e, vsum_epoch_test); 
    xlabel("Parameter Update") 

    tight_layout()

    fig_name = @strdict ntrain nvalid nepochs e lr lr_step gab_l2 λ α αmin β max_recursion n_hidden L K clip_norm
    safesave(joinpath(save_path, savename(fig_name; digits=6)*"mnist_hint_log.png"), fig); close(fig)
    close(fig)

end

print("done")