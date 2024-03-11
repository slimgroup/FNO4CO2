export relu01, perm_to_tensor, gen_grid, jitter, Patchy, read_velocity_cigs_offsets_as_nc, plot_cig_eval

##### constrain to 0-1 #####
relu01(x::AbstractArray{Float32}) = 1f0.-relu.(1f0.-relu.(x))
relu01(x::AbstractArray{Float64}) = 1f0.-relu.(1f0.-relu.(x))

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

#### Patchy saturation model

function Patchy(sw::AbstractMatrix{T}, vp::AbstractMatrix{T}, rho::AbstractMatrix{T}, phi::AbstractMatrix{T}; bulk_min = 36.6f9, bulk_fl1 = 2.735f9, bulk_fl2 = 0.125f9, ρw = 501.9f0, ρo = 1053.0f0) where T

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

function Patchy(sw::AbstractArray{T, 3}, vp::AbstractMatrix{T}, rho::AbstractMatrix{T}, phi::AbstractMatrix{T}; bulk_min = 36.6f9, bulk_fl1 = 2.735f9, bulk_fl2 = 0.125f9, ρw = 501.9f0, ρo = 1053.0f0) where T

    stack = [Patchy(sw[i,:,:], vp, rho, phi; bulk_min=bulk_min, bulk_fl1=bulk_fl1, bulk_fl2=bulk_fl2, ρw = ρw, ρo=ρo) for i = 1:size(sw,1)]
    return [stack[i][1] for i = 1:size(sw,1)], [stack[i][2] for i = 1:size(sw,1)]
end

function plot_cig_eval(modelConfig, plot_path, figname, x_plot, y_plot, y_predict; use_nz=false, additional=Dict{String,Any}())

    params = Config.get_parameters()

    offset_start = params["offset_start"]
    offset_end = params["offset_end"]
    d = params["d"]
    n = [modelConfig.nx, modelConfig.ny]
    n_offsets = params["n_offsets"]

    # Reshape the data to fit the model configuration
    x_plot = reshape(x_plot, (modelConfig.nc_in, modelConfig.nt, modelConfig.nx, modelConfig.ny, modelConfig.nz))
    y_plot = reshape(y_plot, (modelConfig.nc_out, modelConfig.nt, modelConfig.nx, modelConfig.ny, modelConfig.nz))
    y_predict = reshape(y_predict, (modelConfig.nc_out, modelConfig.nt, modelConfig.nx, modelConfig.ny, modelConfig.nz))
    
    perturbed_model = x_plot[1, 1, :, :, 1]
    init_background_model = x_plot[2, 1, :, :, 1]

    input_CIG = use_nz ? x_plot[3, 1, :, :, :] : permutedims(x_plot[3:(3+n_offsets-1), 1, :, :, 1], [2, 3, 1])
    output_CIG = use_nz ? y_predict[1, 1, :, :, :] : permutedims(y_predict[:, 1, :, :, 1], [2, 3, 1])
    true_CIG = use_nz ? y_plot[1, 1, :, :, :] : permutedims(y_plot[:, 1, :, :, 1], [2, 3, 1])

    PyPlot.rc("figure", titlesize=40)
    PyPlot.rc("font", family="serif"); PyPlot.rc("xtick", labelsize=40); PyPlot.rc("ytick", labelsize=40)
    PyPlot.rc("axes", labelsize=40)     # Default fontsize for x and y labels
    PyPlot.rc("axes", titlesize=40)     # Default fontsize for titles
    
    fig, axs = subplots(2, 12, figsize=(120,10), gridspec_kw = Dict("width_ratios" => [4, 4, 1, 4, 4, 1, 4, 1, 4, 1, 4, 1], "height_ratios" => [1, 3]))

    axs[1, 1].remove()
    axs[1, 4].remove()

    plot_velocity_model_helper(perturbed_model, n, d, axs[2, 4])
    plot_velocity_model_helper(init_background_model, n, d, axs[2, 1])

    plot_cig_helper(input_CIG, n, d, offset_start, offset_end, n_offsets, axs[:, 2:3])
    plot_cig_helper(true_CIG, n, d, offset_start, offset_end, n_offsets, axs[:, 5:6])
    plot_cig_helper(output_CIG, n, d, offset_start, offset_end, n_offsets, axs[:, 7:8])
    plot_cig_helper(5f0 .* abs.(true_CIG - output_CIG), n, d, offset_start, offset_end, n_offsets, axs[:, 9:10])
    plot_cig_helper(abs.(fftshift(fft(true_CIG - output_CIG))), n, d, offset_start, offset_end, n_offsets, axs[:, 11:12])

    suptitle("Model and CIG Comparison", fontsize=20)
    
    # figname = _getFigname(trainConfig, additional)
    tight_layout()

    safesave(joinpath(plot_path, savename(figname; digits=6) * "_DFNO_CIG_fitting.png"), fig) #, bbox_inches="tight", dpi=300)
    close(fig)
end

function plot_velocity_model_helper(x, n, d, ax)
    # Assume that vmin and vmax are computed similarly to the plot_cig_helper function
    vmin = quantile(vec(x), 0.05)  # 5th percentile
    vmax = quantile(vec(x), 0.95)  # 95th percentile
    extentfull = (0f0, (n[1]-1)*d[1], (n[2]-1)*d[2], 0f0)
    cax = ax.imshow(x', vmin=vmin, vmax=vmax, extent=extentfull, aspect="auto")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Z [m]")
end

function plot_cig_helper(cig, n, d, offset_start, offset_end, n_offsets, axs)
    y = reshape(cig, n[1], n[2], n_offsets)
    
    ### X, Z position in km
    xpos = 3.6f3
    zpos = 2.7f3
    xgrid = Int(round(xpos / d[1]))
    zgrid = Int(round(zpos / d[2]))
    
    # Adjust the spacing between the plots
    subplots_adjust(hspace=0.0, wspace=0.0)
    
    vmin1, vmax1 = (-1, 1) .* quantile(abs.(vec(y[:,zgrid,:,1])), 0.99)
    vmin2, vmax2 = (-1, 1) .* quantile(abs.(vec(y[:,:,div(n_offsets,2)+1,1])), 0.88)
    vmin3, vmax3 = (-1, 1) .* quantile(abs.(vec(y[xgrid,:,:,1])), 0.999)
    sca(axs[1, 1])
    
    # Top left subplot
    axs[1, 1].imshow(y[:,zgrid,:,1]', aspect="auto", cmap="gray", interpolation="none",vmin=vmin1, vmax=vmax1,
        extent=(0f0, (n[1]-1)*d[1], offset_start, offset_end))
    axs[1, 1].set_ylabel("Offset [m]", fontsize=40)
    axs[1, 1].set_xticklabels([])
    axs[1, 1].set_xlabel("")
    hlines(y=0, colors=:b, xmin=0, xmax=(n[1]-1)*d[1], linewidth=3)
    vlines(x=xpos, colors=:b, ymin=offset_start, ymax=offset_end, linewidth=3)
    
    # Bottom left subplot
    sca(axs[2, 1])
    axs[2, 1].imshow(y[:,:,div(n_offsets,2)+1,1]', aspect="auto", cmap="gray", interpolation="none",vmin=vmin2, vmax=vmax2,
    extent=(0f0, (n[1]-1)*d[1], (n[2]-1)*d[2], 0f0))
    axs[2, 1].set_xlabel("X [m]", fontsize=40)
    axs[2, 1].set_ylabel("Z [m]", fontsize=40)
    axs[2, 1].set_xticks([0, 2000, 4000, 6000])
    axs[2, 1].set_xticklabels(["0", "2000", "4000", "6000"])
    axs[2, 1].set_yticks([1000, 2000, 3000])
    axs[2, 1].set_yticklabels(["1000", "2000", "3000"])
    
    # axs[2, 2].get_shared_x_axes().join(axs[1, 1], axs[2, 1])
    vlines(x=xpos, colors=:b, ymin=0, ymax=(n[2]-1)*d[2], linewidth=3)
    hlines(y=zpos, colors=:b, xmin=0, xmax=(n[1]-1)*d[1], linewidth=3)
    
    # Top right subplot
    axs[1, 2].set_visible(false)
    
    # Bottom right subplot
    sca(axs[2, 2])
    axs[2, 2].imshow(y[xgrid,:,:,1], aspect="auto", cmap="gray", interpolation="none",vmin=vmin3, vmax=vmax3,
    extent=(offset_start, offset_end, (n[2]-1)*d[2], 0f0))
    axs[2, 2].set_xlabel("Offset [m]", fontsize=40)
    # Share y-axis with bottom left
    # axs[2, 2].get_shared_y_axes().join(axs[2, 2], axs[2, 1])
    axs[2, 2].set_yticklabels([])
    axs[2, 2].set_ylabel("")
    vlines(x=0, colors=:b, ymin=0, ymax=(n[2]-1)*d[2], linewidth=3)
    hlines(y=zpos, colors=:b, xmin=offset_end, xmax=offset_start, linewidth=3)
end

function _getFigname(config::DFNO_3D.TrainConfig, additional::Dict)
    nbatch = config.nbatch
    epochs = config.epochs
    ntrain = size(config.x_train, 3)
    nvalid = size(config.x_valid, 3)
    
    figname = @strdict nbatch epochs ntrain nvalid
    return merge(additional, figname)
end

function read_velocity_cigs_offsets_as_nc(path::String, modelConfig::DFNO_3D.ModelConfig; ntrain::Int, nvalid::Int)

    params = Config.get_parameters()

    offset_start = params["read_offset_start"]
    offset_end = params["read_offset_end"]

    # Assumption that x is (nx, nz, 1, n). x0 is (nx, nz). CIG0 is (nh, nx, nz). CIG is (nh, nx, nz, 1, n)
    function read_x_tensor(file_name, key, indices)
        data = nothing
        h5open(file_name, "r") do file
            x_data = file[key[1]]
            x0_data = file[key[2]]
            cig0_data = file[key[3]]
            offsets = modelConfig.nc_in - 6 # - indices - 2 velocity models

            # Read proper indices of x and x0. NOTE: Disclude 3 because no z = 1 for background
            x = x_data[indices[1], indices[2], 1, indices[4]]
            x0 = x0_data[indices[1], indices[2]]
            cig0 = cig0_data[offset_start:offset_end, indices[1], indices[2]]

            # Reshape to prepare for augmentation
            x = reshape(x, :, map(range -> length(range), indices[1:4])...)
            x0 = reshape(x0, :, map(range -> length(range), indices[1:3])..., 1)
            cig0 = reshape(cig0, :, map(range -> length(range), indices[1:3])..., 1)

            x0 = repeat(x0, outer=[1, 1, 1, 1, length(indices[4])])
            cig0 = repeat(cig0 ./ 2f3, outer=[1, 1, 1, 1, length(indices[4])])

            # Concat along dimension 1
            data = cat(x, x0, cig0, dims=1)
        end

        # data_channels * nx * ny * nz * nt * n = data_channels * nx * nz * nh * 1 * n
        return data
    end
    
    function read_y_tensor(file_name, key, indices)
        data = nothing
        h5open(file_name, "r") do file
            offsets = modelConfig.nc_in - 6 # - indices - 2 velocity models
            cigs_data = file[key]
            data = cigs_data[offset_start:offset_end, indices[1], indices[2], indices[4], indices[5]] # first dim is offsets as channel, dim 4 which is t = 1:1
        end

        # channels * nx * ny * nz * nt * n = channels * nx * nz * nh * 1 * n
        return reshape(data ./ 2f3, :, map(range -> length(range), indices[1:5])...)
    end

    dataConfig = DFNO_3D.DataConfig(modelConfig=modelConfig, 
                                    ntrain=ntrain, 
                                    nvalid=nvalid, 
                                    perm_file=path,
                                    conc_file=path,
                                    perm_key=["all_xs", "x", "cig0"],
                                    conc_key="all_cigs")

    return DFNO_3D.loadDistData(dataConfig, dist_read_x_tensor=read_x_tensor, dist_read_y_tensor=read_y_tensor)
end
