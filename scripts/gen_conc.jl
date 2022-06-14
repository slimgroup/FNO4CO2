#### Generate time-varying CO2 concentration for each permeability
## Author: Ziyi Yin, ziyi.yin@gatech.edu

using DrWatson
@quickactivate "FNO4CO2"

using Seis4CCS.FlowSimulation
using MAT

# Hyperparameter for flow simulation
n = (64, 64)    # num of cells
d = (15.0, 15.0)        # meter
nt = 50       # number of time steps
dt = 20.0       # day

grid_ = comp_grid(n, d, 10.0, nt, dt);

ϕ = 0.25 .* ones(n)
qw = zeros(nt, n[1], n[2])
qw[:,3,32] .= 0.005
qo = zeros(nt, n[1], n[2])
qo[:,62,32] .= -0.005

nsamples = 1200
perm = matread("perm_gridspacing$(d[1]).mat")["perm"];
conc = zeros(Float32,nt+1,n[1],n[2],nsamples);

for i = 1:nsamples
    println("sample", i)
    @time conc[:,:,:,i] = flow(perm[:,:,i], ϕ, qw, qo, grid_)[1];
end

# Define raw data directory
mkpath(datadir("training-data"))
perm_path = datadir("training-data", "perm_gridspacing15.0.mat")
conc_path = datadir("training-data", "conc_gridspacing15.0.mat")

matwrite(conc_path, Dict(
	"conc" => conc,
); compress = true)