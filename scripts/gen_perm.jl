#### Generate a dataset of permeability models
## Author: Ziyi Yin, ziyi.yin@gatech.edu

using DrWatson
@quickactivate "FNO4CO2"

using PyPlot
using Random, MAT, Images, JLD2
using Distributions
using LinearAlgebra
using Images

Random.seed!(1234)


num_sample = 10000
n = (64,64)
d = (15.0, 15.0)

perm = zeros(Float32,n[1],n[2],num_sample)

function gaussian_kernel(xa,xy;theta0=1,delta=1,cons=1f-5)
    return theta0*exp.(-(xa.-xy).^2f0*1f0/delta^2f0)+theta0*cons*I
end

nb_of_samples = n[1]
nb_of_functions = 2

theta0 = 5
delta = 25
cons = 1f-5

X = convert(Array{Float32},reshape(range(4f0,stop=n[1]*4f0,length=n[1]),:,1))
Cova = gaussian_kernel(X,X',theta0=theta0,delta=delta,cons=cons)

figure();
for i = 1:10
line1 = rand(MvNormal(zeros(Float32,n[1]),Cova))
plot(line1)
end
title("sample from gaussian process")

for i = 1:num_sample
    #perm_top = 20f0*rand(Float32)+15f0   # 10-20 MD
    perm_top = 20f0
    #perm_mid = 80f0*rand(Float32)+70f0    # 60-180 MD
    perm_mid = 120f0
    #perm_down = 20f0*rand(Float32)+15f0   # 10-20 MD
    perm_down = deepcopy(perm_top)
    cap_start = Int(round(rand(Float32)*0.05*n[2]+0.37*n[2]))
    cap_end = Int(round(rand(Float32)*0.05*n[2]+0.55*n[2]))

    cap_ceil = Int.(round.(rand(MvNormal(zeros(Float32,n[1]), Cova)) .+ cap_start))
    cap_floor = Int.(round.(rand(MvNormal(zeros(Float32,n[1]), Cova)) .+ cap_end))

    perm_ = zeros(Float32, n)

    for j = 1:n[1]
        perm_[j,1:cap_ceil[j]] .= perm_top
        perm_[j,cap_ceil[j]+1:cap_floor[j]] .= perm_mid
        perm_[j,cap_floor[j]+1:end] .= perm_down
    end

    perm[:,:,i] = deepcopy(perm_)
end

figure(figsize=(12,12));
for i = 1:9
    subplot(3,3,i)
    imshow(perm[:,:,i]',vmin=20,vmax=120)
    colorbar()
end
suptitle("millidarcy -- samples")

figure(figsize=(12,12));
for i = 1:9
    subplot(3,3,i)
    imshow(perm[:,:,end-i]',vmin=20,vmax=120)
    colorbar()
end
suptitle("millidarcy -- samples")

# Define raw data directory
mkpath(datadir("training-data"))
perm_path = datadir("training-data", "perm_gridspacing15.0.mat")

param_dict = @strdict num_sample n d theta0 delta cons perm
@tagsave(
    datadir("training-data", savename(param_dict, "jld2"; digits=6)),
    param_dict;
    safe = true
)

perm = perm[:,:,1:1200]
matwrite(perm_path, Dict(
	"perm" => perm,
); compress = true)