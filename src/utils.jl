export relu01, perm_to_tensor, gen_grid, jitter, Patchy

##### constrain to 0-1 #####
relu01(x::AbstractArray{Float32}) = 1f0.-relu.(1f0.-relu.(x))

##### generate grid #####

## x,y
function gen_grid(n::Tuple{Integer, Integer},d::Tuple{Float32, Float32})
    grid = zeros(Float32,n[1],n[2],2)   
    grid[:,:,1] = repeat(reshape(collect(range(d[1],stop=n[1]*d[1],length=n[1])), :, 1)',n[2])' # x
    grid[:,:,2] = repeat(reshape(collect(range(d[2],stop=n[2]*d[2],length=n[2])), 1, :),n[1])   # z
    return grid
end

## x,y,t
function gen_grid(n::Tuple{Integer, Integer},d::Tuple{Float32, Float32},nt::Int,dt::Float32)
    tsample = [(i-1)*dt for i = 1:nt]
    return gen_grid(n, d, tsample)
end

function gen_grid(n::Tuple{Integer, Integer},d::Tuple{Float32, Float32},tsample::Vector{Float32})
    nt = length(tsample)
    grid = zeros(Float32,n[1],n[2],nt,3)
    for i = 1:nt     
        grid[:,:,i,1] = repeat(reshape(collect(range(d[1],stop=n[1]*d[1],length=n[1])), :, 1)',n[2])' # x
        grid[:,:,i,2] = repeat(reshape(collect(range(d[2],stop=n[2]*d[2],length=n[2])), 1, :),n[1])   # z
        grid[:,:,i,3] .= tsample[i]   # t
    end
    return grid
end

##### turn 2D permeability to a tensor input for FNO #####

function perm_to_tensor(x_perm::AbstractMatrix{Float32},grid::Array{Float32,4},AN::ActNorm)
    # input nx*ny, output nx*ny*nt*4*1
    nx, ny, nt, _ = size(grid)
    return cat(reshape(cat([AN(reshape(x_perm, nx, ny, 1, 1))[:,:,1,1] for i = 1:nt]..., dims=3), nx, ny, nt, 1, 1),
    reshape(grid, nx, ny, nt, 3, 1), dims=4)
end

function perm_to_tensor(x_perm::AbstractMatrix{Float32},grid::Array{Float32,4})
    # input nx*ny, output nx*ny*nt*4*1
    nx, ny, nt, _ = size(grid)
    return cat(reshape(cat([reshape(x_perm, nx, ny, 1, 1)[:,:,1,1] for i = 1:nt]..., dims=3), nx, ny, nt, 1, 1),
    reshape(grid, nx, ny, nt, 3, 1), dims=4)
end

perm_to_tensor(x_perm::AbstractArray{Float32,3},grid::Array{Float32,4},AN::ActNorm) = cat([perm_to_tensor(x_perm[:,:,i],grid,AN) for i = 1:size(x_perm,3)]..., dims=5)

perm_to_tensor(x_perm::AbstractArray{Float32,3},grid::Array{Float32,4}) = cat([perm_to_tensor(x_perm[:,:,i],grid) for i = 1:size(x_perm,3)]..., dims=5)

#### seismic Utilities

function jitter(nsrc::Int, nssample::Int)
    npatch = Int(nsrc/nssample)
    return rand(1:npatch, nssample) .+ convert(Vector{Int},0:npatch:(nsrc-1))
end

function Patchy(sw::AbstractMatrix{T1}, vp::AbstractMatrix{T}, rho::AbstractMatrix{T}, phi::AbstractMatrix{T}; bulk_min = 36.6f9, bulk_fl1 = 2.735f9, bulk_fl2 = 0.125f9, ρw = 501.9f0, ρo = 1053.0f0) where {T1, T}

    ### works for channel problem
    vs = vp./sqrt(3f0)
    sw = T.(sw)
    bulk_sat1 = rho .* (vp.^2f0 - 4f0/3f0 .* vs.^2f0)
    shear_sat1 = rho .* (vs.^2f0)

    patch_temp = bulk_sat1 ./(bulk_min .- bulk_sat1) - 
    bulk_fl1 ./ phi ./ (bulk_min .- bulk_fl1) + 
    bulk_fl2 ./ phi ./ (bulk_min .- bulk_fl2)

    bulk_sat2 = bulk_min./(1f0./patch_temp .+ 1f0)

    bulk_new = 1f0./( (1f0.-sw)./(bulk_sat1+4f0/3f0*shear_sat1) 
    + sw./(bulk_sat2+4f0/3f0*shear_sat1) ) - 4f0/3f0*shear_sat1

    rho_new = rho + phi .* sw * (ρw - ρo)

    Vp_new = sqrt.((bulk_new+4f0/3f0*shear_sat1)./rho_new)
    return Vp_new, rho_new

end

function Patchy(sw::AbstractArray{T1, 3}, vp::AbstractMatrix{T}, rho::AbstractMatrix{T}, phi::AbstractMatrix{T}; bulk_min = 36.6f9, bulk_fl1 = 2.735f9, bulk_fl2 = 0.125f9, ρw = 501.9f0, ρo = 1053.0f0) where {T1, T}

    stack = [Patchy(sw[i,:,:], vp, rho, phi; bulk_min=36.6f9, bulk_fl1=bulk_fl1, bulk_fl2=bulk_fl2, ρw = ρw, ρo=ρo) for i = 1:size(sw,1)]
    return [stack[i][1] for i = 1:size(sw,1)], [stack[i][2] for i = 1:size(sw,1)]
end

function Patchy(sw::AbstractMatrix{T1}, vp::AbstractMatrix{T}, rho::AbstractMatrix{T}, phi::AbstractMatrix{T}, d::Tuple{T, T}; bulk_fl1 = 2.735f9, bulk_fl2 = 0.125f9,ρw = 7.766f2, ρo = 1.053f3) where {T1, T}

    ### works for Compass 2D model
    n = size(sw)
    capgrid = Int(round(50f0/d[2]))
    vp = vp * 1f3
    vs = vp ./ sqrt(3f0)
    idx_wb = maximum(find_water_bottom(vp.-minimum(vp)))
    idx_ucfmt = find_water_bottom((vp.-3500f0).*(vp.>3500f0))

    bulk_sat1 = rho .* (vp.^2f0 - 4f0/3f0 .* vs.^2f0) * 1f3
    shear_sat1 = rho .* (vs.^2f0) * 1f3

    bulk_min = zeros(Float32,size(bulk_sat1))

    bulk_min[findall(vp.>=3500f0)] .= 5f10   # mineral bulk moduli
    bulk_min[findall(vp.<3500f0)] .= 1.2f0 * bulk_sat1[findall(vp.<3500f0)] # mineral bulk moduli

    patch_temp = bulk_sat1 ./(bulk_min .- bulk_sat1) - bulk_fl1 ./ phi ./ (bulk_min .- bulk_fl1) + bulk_fl2 ./ phi ./ (bulk_min .- bulk_fl2)

    for i = 1:n[1]
        patch_temp[i,idx_ucfmt[i]-capgrid:idx_ucfmt[i]-1] = patch_temp[i,idx_ucfmt[i]:idx_ucfmt[i]+capgrid-1]
    end

    bulk_sat2 = bulk_min./(1f0./patch_temp .+ 1f0)
    bulk_sat2[findall(bulk_sat2-bulk_sat1.>0)] = bulk_sat1[findall(bulk_sat2-bulk_sat1.>0)]

    bulk_new = (bulk_sat1+4f0/3f0*shear_sat1).*(bulk_sat2+4f0/3f0*shear_sat1) ./( (1f0.-T.(sw)).*(bulk_sat2+4f0/3f0*shear_sat1) 
    + T.(sw).*(bulk_sat1+4f0/3f0*shear_sat1) ) - 4f0/3f0*shear_sat1

	bulk_new[:,1:idx_wb] = bulk_sat1[:,1:idx_wb]
    bulk_new[findall(sw.==0)] = bulk_sat1[findall(sw.==0)]
    rho_new = rho + phi .* T.(sw) * (ρw - ρo) / 1f3
    rho_new[findall(sw.==0)] = rho[findall(sw.==0)]
    Vp_new = sqrt.((bulk_new+4f0/3f0*shear_sat1)./rho_new/1f3)
    Vp_new[findall(sw.==0)] = vp[findall(sw.==0)]

    return Vp_new/1f3, rho_new
end
function Patchy(sw::AbstractArray{T1, 3}, vp::AbstractMatrix{T}, rho::AbstractMatrix{T}, phi::AbstractMatrix{T}, d::Tuple{T, T}; bulk_fl1 = 2.735f9, bulk_fl2 = 0.125f9,ρw = 7.766f2, ρo = 1.053f3) where {T1, T}

    stack = [Patchy(sw[i,:,:], vp, rho, phi, d; bulk_fl1=bulk_fl1, bulk_fl2=bulk_fl2, ρw = ρw, ρo=ρo) for i = 1:size(sw,1)]
    return [stack[i][1] for i = 1:size(sw,1)], [stack[i][2] for i = 1:size(sw,1)]
end
