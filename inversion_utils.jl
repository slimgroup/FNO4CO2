mutable struct compgrid
    d :: Tuple
    n :: Tuple
end

##### Rock physics

function Patchy(sw::AbstractMatrix{Float32}, vp::AbstractMatrix{Float32}, vs::AbstractMatrix{Float32}, rho::AbstractMatrix{Float32}, phi::AbstractMatrix{Float32}; bulk_min = 36.6f9, bulk_fl1 = 2.735f9, bulk_fl2 = 0.125f9, ρw = 501.9f0, ρo = 1053.0f0)

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
    Vs_new = sqrt.((shear_sat1)./rho_new)

    return Vp_new, Vs_new, rho_new
end

function Patchy(sw::AbstractArray{Float32, 3}, vp::AbstractMatrix{Float32}, vs::AbstractMatrix{Float32}, rho::AbstractMatrix{Float32}, phi::AbstractMatrix{Float32}; bulk_min = 36.6f9, bulk_fl1 = 2.735f9, bulk_fl2 = 0.125f9, ρw = 501.9f0, ρo = 1053.0f0)

    Vp_new = similar(sw)
    Vs_new = similar(sw)
    rho_new = similar(sw)
    for i = 1:size(sw, 3)
        Vp_new[:,:,i], Vs_new[:,:,i], rho_new[:,:,i] = Patchy(sw[:,:,i], vp, vs, rho, phi; bulk_min = 36.6f9, bulk_fl1 = 2.735f9, bulk_fl2 = 0.125f9, ρw = 501.9f0, ρo = 1053.0f0)
    end
    return Vp_new, Vs_new, rho_new
end

function sample_src(d_obs, nsrc, rand_ns)
    datalength = Int(length(d_obs)/nsrc)
    return vcat([d_obs[(rand_ns[i]-1)*datalength+1:rand_ns[i]*datalength] for i = 1:length(rand_ns)]...)
end


function perm_to_tensor(x_perm::AbstractMatrix{Float32},nt::Int,grid::Array,dt::Float32)
    # input nx*ny, output nx*ny*nt*4*1
    nx, ny = size(x_perm)
    x1 = reshape(x_perm,nx,ny,1,1,1)
    x2 = cat([x1 for i = 1:nt]...,dims=3)
    grid_1 = cat([reshape(grid[:,:,1],nx,ny,1,1,1) for i = 1:nt]...,dims=3)
    grid_2 = cat([reshape(grid[:,:,2],nx,ny,1,1,1) for i = 1:nt]...,dims=3)
    grid_t = cat([(i-1)*dt*ones(Float32,nx,ny,1,1,1) for i = 1:nt]...,dims=3)
    x_out = cat(x2,grid_1,grid_2,grid_t,dims=4)
    return x_out
end

function perm_to_tensor(x_perm::AbstractMatrix{Float32},tsample::Vector{Float32},grid::Array)
    # input nx*ny, output nx*ny*nt*4*1
    nx, ny = size(x_perm)
    nt = length(tsample)
    x1 = reshape(x_perm,nx,ny,1,1,1)
    x2 = cat([x1 for i = 1:nt]...,dims=3)
    grid_1 = cat([reshape(grid[:,:,1],nx,ny,1,1,1) for i = 1:nt]...,dims=3)
    grid_2 = cat([reshape(grid[:,:,2],nx,ny,1,1,1) for i = 1:nt]...,dims=3)
    grid_t = cat([tsample[i]*ones(Float32,nx,ny,1,1,1) for i = 1:nt]...,dims=3)
    x_out = cat(x2,grid_1,grid_2,grid_t,dims=4)
    return x_out
end

function jitter(nsrc::Int, nssample::Int)
    npatch = Int(nsrc/nssample)
    return rand(1:npatch, nssample) .+ convert(Vector{Int},0:npatch:(nsrc-1))
end

function ContJitter(l::Number, num::Int)
    #l = length, num = number of samples
    interval_width = l/num
    interval_center = range(interval_width/2, stop = l-interval_width/2, length=num)
    randomshift = interval_width .* rand(Float32, num) .- interval_width/2

    return interval_center .+ randomshift
end

function SetPrj()
    
    comp_grid = compgrid(d,n)
    
    options          = PARSDMM_options()
    options.FL       = Float32
    options.feas_tol = 0.001f0
    options.evol_rel_tol = 0.0001f0
    set_zero_subnormals(true)
    
    constraint = Vector{SetIntersectionProjection.set_definitions}()
    
    # TV:
    TV = get_TD_operator(comp_grid,"TV",options.FL)[1]
    m_min     = 0.0
    m_max     = 1.2f0*norm(TV*vec(encode(x_true)),1)
    set_type  = "l1"
    TD_OP     = "TV"
    app_mode  = ("matrix","")
    custom_TD_OP = ([],false)
    push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))
    
    BLAS.set_num_threads(12)
    (P_sub,TD_OPS,set_Prop) = setup_constraints(constraint,comp_grid,options.FL)
    (TD_OPS,AtA,l,y) = PARSDMM_precompute_distribute(TD_OPS,set_Prop,comp_grid,options)
    
    function prj(x; vmin=10f0, vmax=130f0)
        @time (x_perm1,log_PARSDMM) = PARSDMM(vec(x),AtA,TD_OPS,set_Prop,P_sub,comp_grid,options);
        xtv = reshape(x_perm1,n)
        y = decode(xtv)
        z = max.(min.(y,vmax),vmin)
        return encode(z)
    end

    return prj
    
end

function S(x::AbstractMatrix{Float32})
    return decode(y_normalizer,NN(perm_to_tensor(x,nt,grid,dt)))[:,:,survey_indices,1]
end

function R(c::AbstractArray{Float32,3})
    return [(Patchy(c[:,:,i]',vp,vs,rho,phi))[1] for i = 1:nv]
end

function (F::Array{Forward{judiPDEfull{Float32,Float32}},1})(v::Vector{Matrix{Float32}})
    return [F[i]((1f3 ./ v[i]).^2f0) for i = 1:nv]
end