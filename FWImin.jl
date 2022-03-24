using PyPlot
using LinearAlgebra
using CUDA
using JLD2
using LineSearches
using JUDI
using Random

JLD2.@load "result/mm0.jld2" m m0

n = (64, 64)
d = (15f0, 15f0)
o = (0f0, 0f0)

nsrc = 16
nrec = Int(round((n[2]-1)*d[2]))

model = Model(n, d, o, m; nb = 80)
model0 = Model(n, d, o, m0; nb = 80)

timeS = timeR = 750f0
dtS = dtR = 1f0
ntS = Int(floor(timeS/dtS))+1
ntR = Int(floor(timeR/dtR))+1

xsrc = convertToCell(range(1*d[1],stop=1*d[1],length=nsrc))
ysrc = convertToCell(range(0f0,stop=0f0,length=nsrc))
zsrc = convertToCell(range(d[2],stop=(n[2]-1)*d[2],length=nsrc))

xrec = range((n[1]-1)*d[1],stop=(n[1]-1)*d[1], length=nrec)
yrec = 0f0
zrec = range(d[2],stop=(n[2]-1)*d[2],length=nrec)

srcGeometry = Geometry(xsrc, ysrc, zsrc; dt=dtS, t=timeS)
recGeometry = Geometry(xrec, yrec, zrec; dt=dtR, t=timeR, nsrc=nsrc)

f0 = 0.02f0     # kHz
wavelet = ricker_wavelet(timeS, dtS, f0)
q = judiVector(srcGeometry, wavelet)

ntComp = get_computational_nt(srcGeometry, recGeometry, model0)
info = Info(prod(n), nsrc, ntComp)

opt = Options()
Pr = judiProjection(info, recGeometry)
Ps = judiProjection(info, srcGeometry)

F = Pr*judiModeling(info, model; options=opt)*Ps'
F0 = Pr*judiModeling(info, model0; options=opt)*Ps'
d_obs = F*q

ls = BackTracking(order=3, iterations=10)
vmin = minimum((1f0/6.5f0)^2f0)
vmax = maximum((1f0/1.3f0)^2f0)
function proj(x)
    z = max.(min.(x,vmax),vmin)
    return z
end

figure();imshow(m');title("ground truth")

niterations = 50
hisloss = zeros(Float32, niterations)
figv, axv = subplots(nrows=1,ncols=1,figsize=(20,12))
figgrad, axgrad = subplots(nrows=1,ncols=1,figsize=(20,12))

batchsize = nsrc # work with all source now
for j=1:niterations

    # get fwi objective function value and gradient
    i = randperm(d_obs.nsrc)[1:batchsize]
    fval, gradient = fwi_objective(model0, q[i], d_obs[i])
    p = -gradient/norm(gradient, Inf)
    
    println("FWI iteration no: ",j,"; function value: ",fval)
    hisloss[j] = fval

    mnow = deepcopy(model0.m)

    # linesearch
    function ϕ(α)
        F0.model.m .= proj(mnow .+ α * p)
        misfit = .5*norm(F0[i]*q[i] - d_obs[i])^2f0
        @show α, misfit
        return misfit
    end
    step, fval = ls(ϕ, 1f-1, fval, dot(gradient, p))

    # Update model and bound projection
    model0.m .= proj(mnow .+ step .* p)
    axv.imshow(model0.m');axv.set_title("m after $j iterations")
    axgrad.imshow(gradient');axgrad.set_title("g after $j iterations")
end
