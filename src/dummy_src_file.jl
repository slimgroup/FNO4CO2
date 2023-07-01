"""
    dummy_project_function(x, y) → z
Dummy function for illustration purposes.
Performs operation:
```math
z = x + y
```
"""
function dummy_project_function(x, y)
    return x + y
end

import FNO4CO2.perm_to_tensor
function perm_to_tensor(x_perm::AbstractMatrix{Float32},grid::AbstractArray{Float32,4},AN::ActNorm)
    # input nx*ny, output nx*ny*nt*4*1
    nx, ny, nt, _ = size(grid)
    return cat(reshape(cat([AN(reshape(x_perm, nx, ny, 1, 1))[:,:,1,1] for i = 1:nt]..., dims=3), nx, ny, nt, 1, 1),
    reshape(grid, nx, ny, nt, 3, 1), dims=4)
end

function perm_to_tensor(x_perm::AbstractMatrix{Float32},grid::AbstractArray{Float32,4})
    # input nx*ny, output nx*ny*nt*4*1
    nx, ny, nt, _ = size(grid)
    return cat(reshape(cat([reshape(x_perm, nx, ny, 1, 1)[:,:,1,1] for i = 1:nt]..., dims=3), nx, ny, nt, 1, 1),
    reshape(grid, nx, ny, nt, 3, 1), dims=4)
end

perm_to_tensor(x_perm::AbstractArray{Float32,3},grid::AbstractArray{Float32,4},AN::ActNorm) = cat([perm_to_tensor(x_perm[:,:,i],grid,AN) for i = 1:size(x_perm,3)]..., dims=5)

perm_to_tensor(x_perm::AbstractArray{Float32,3},grid::AbstractArray{Float32,4}) = cat([perm_to_tensor(x_perm[:,:,i],grid) for i = 1:size(x_perm,3)]..., dims=5)

function prjz(z::AbstractArray{T}; α=one(T)) where T
    znorm = norm(z)
    gaussian_norm = α * T(sqrt(length(z)))
    if znorm <= gaussian_norm
        return z
    else
        return z/znorm * gaussian_norm
    end
end

#### Patchy saturation model

function Patchy(sw::AbstractMatrix{T}, vp::AbstractMatrix{T}, rho::AbstractMatrix{T}, phi::AbstractMatrix{T};
    bulk_min::T=T(36.6f9), bulk_fl1::T=T(2.735f9), bulk_fl2::T=T(0.125f9), ρw::T=T(501.9f0), ρo::T=T(1053.0f0)) where T

    ### works for channel problem
    vs = vp./sqrt(3f0)
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

function Patchy(sw::AbstractArray{T, 3}, vp::AbstractMatrix{T}, rho::AbstractMatrix{T}, phi::AbstractMatrix{T};
    bulk_min::T=T(36.6f9), bulk_fl1::T=T(2.735f9), bulk_fl2::T=T(0.125f9), ρw::T=T(501.9f0), ρo::T=T(1053.0f0)) where T

    stack = [Patchy(sw[i,:,:], vp, rho, phi; bulk_min=bulk_min, bulk_fl1=bulk_fl1, bulk_fl2=bulk_fl2, ρw = ρw, ρo=ρo) for i = 1:size(sw,1)]
    return [stack[i][1] for i = 1:size(sw,1)], [stack[i][2] for i = 1:size(sw,1)]
end

function Patchy(sw::Vector{Matrix{T}}, vp::Matrix{T}, rho::Matrix{T}, phi::Matrix{T};
    bulk_min::T=T(36.6f9), bulk_fl1::T=T(2.735f9), bulk_fl2::T=T(0.125f9), ρw::T=T(700f0), ρo::T=T(1000.0f0)) where T

    stack = [Patchy(sw[i], vp, rho, phi; bulk_min=bulk_min, bulk_fl1=bulk_fl1, bulk_fl2=bulk_fl2, ρw = ρw, ρo=ρo) for i = 1:size(sw,1)]
    return [stack[i][1] for i = 1:size(sw,1)], [stack[i][2] for i = 1:size(sw,1)]
end

### box
box_logK(x::AbstractArray{T}) where T = max.(min.(x,T(-29.684404930981383)),T(-32.24935428844292))
box_co2(x::AbstractArray{T}) where T = max.(min.(x,T(0.9)),T(0))
box_co2(x::AbstractVector) = [box_co2(x[i]) for i = 1:length(x)]
box_v(x::AbstractMatrix{T}) where T = max.(min.(x,T(3501)),T(3200))
box_v(x::AbstractVector) = [box_v(x[i]) for i = 1:length(x)]
box_K(x::AbstractArray{T}) where T = max.(min.(x,T(130f0)),T(10f0))
