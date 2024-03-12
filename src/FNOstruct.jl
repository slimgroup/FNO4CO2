export Net2d, Net3d

### Spectral convolution
mutable struct SpectralConv{T,N}
    weights::AbstractArray{T,N}
end

@Flux.functor SpectralConv

# Constructor
function SpectralConv(in_channels::Int64, out_channels::Int64, modes::Vector{Int64}; DT::DataType=ComplexF32)
    scale = 1f0 / (in_channels * out_channels)
    weights = scale*rand(DT, modes..., in_channels, out_channels, 2^(length(modes)-1))
    gpu_flag && (weights = weights |> gpu)
    return SpectralConv(weights)
end

function compl_mul(x::AbstractArray{T,4}, y::AbstractArray{T,4}) where T
    # complex multiplication
    # x in (modes1, modes2, input channels, batchsize)
    # y in (modes1, modes2, input channels, output channels)
    # output in (modes1,modes2,output channles,batchsize)
    x_per = permutedims(x,[4,3,1,2]) # batchsize*in_channels*modes1*modes2
    y_per = permutedims(y,[3,4,1,2]) # in_channels*out_channels*modes1*modes2
    x_resh = reshape(x_per,size(x_per,1),size(x_per,2),:) # batchsize*in_channels*(modes1*modes2)
    y_resh = reshape(y_per,size(y_per,1),size(y_per,2),:) # in_channels*out_channels*(modes1*modes2)
    out_resh = batched_mul(x_resh,y_resh) # batchsize*out_channels*(modes1*modes2)
    out_per = reshape(out_resh,size(out_resh,1),size(out_resh,2),size(x,1),size(x,2)) # batchsize*out_channels*modes1*modes2
    out = permutedims(out_per,[3,4,2,1])
    return out
end

function compl_mul(x::AbstractArray{T, 5}, y::AbstractArray{T, 5}) where T
    # complex multiplication
    # x in (modes1, modes2, modes3, input channels, batchsize)
    # y in (modes1, modes2, modes3, input channels, output channels)
    # output in (modes1,modes2,modes3,output channels,batchsize)
    x_per = permutedims(x,[5,4,1,2,3]) # batchsize*in_channels*modes1*modes2*modes3
    y_per = permutedims(y,[4,5,1,2,3]) # in_channels*out_channels*modes1*modes2*modes3
    x_resh = reshape(x_per,size(x_per,1),size(x_per,2),:) # batchsize*in_channels*(modes1*modes2*modes3)
    y_resh = reshape(y_per,size(y_per,1),size(y_per,2),:) # in_channels*out_channels*(modes1*modes2*modes3)
    out_resh = batched_mul(x_resh,y_resh) # batchsize*out_channels*(modes1*modes2*modes3)
    out_per = reshape(out_resh,size(out_resh,1),size(out_resh,2),size(x,1),size(x,2),size(x,3)) # batchsize*out_channels*modes1*modes2*modes3
    out = permutedims(out_per,[3,4,5,2,1])
    return out
end

function (L::SpectralConv)(x::AbstractArray{T, 4}) where T
    # x in (size_x, size_y, channels, batchsize)
    x_ft = rfft(x,[1,2])
    (modes1, modes2) = size(L.weights)[[1,2]]
    out_ft = cat(cat(compl_mul(x_ft[1:modes1, 1:modes2,:,:], L.weights[:,:,:,:,1]),
                zeros(Complex{T}, modes1, size(x_ft,2)-2*modes2, size(x_ft,3), size(x_ft,4)),
                compl_mul(x_ft[1:modes1, end-modes2+1:end,:,:], L.weights[:,:,:,:,2]),dims=2),
                zeros(Complex{T}, size(x_ft,1)-modes1, size(x_ft,2), size(x_ft,3), size(x_ft,4)),dims=1)
    x = irfft(out_ft, size(x,1),[1,2])
end

function (L::SpectralConv)(x::AbstractArray{T, 5}) where T
    # x in (size_x, size_y, time, channels, batchsize)
    x_ft = rfft(x,[1,2,3])      ## full size FFT
    (modes1, modes2, modes3) = size(L.weights)[[1,2,3]]

    ### only keep low frequency coefficients
    out_ft = cat(cat(cat(compl_mul(x_ft[1:modes1, 1:modes2, 1:modes3, :,:], L.weights[:,:,:,:,:,1]),
                zeros(Complex{T}, modes1, modes2, size(x_ft,3)-2*modes3, size(x_ft,4), size(x_ft,5)), 
                compl_mul(x_ft[1:modes1, 1:modes2, end-modes3+1:end,:,:], L.weights[:,:,:,:,:,2]),dims=3),
                zeros(Complex{T}, modes1, size(x_ft, 2)-2*modes2, size(x_ft,3), size(x_ft,4), size(x_ft,5)),
                cat(compl_mul(x_ft[1:modes1, end-modes2+1:end, 1:modes3,:,:], L.weights[:,:,:,:,:,3]),
                zeros(Complex{T}, modes1, modes2, size(x_ft,3)-2*modes3, size(x_ft,4), size(x_ft,5)),
                compl_mul(x_ft[1:modes1, end-modes2+1:end, end-modes3+1:end,:,:], L.weights[:,:,:,:,:,4]),dims=3)
                ,dims=2),
                zeros(Complex{T}, size(x_ft,1)-modes1, size(x_ft,2), size(x_ft,3), size(x_ft,4), size(x_ft,5)),dims=1)
    out_ft = irfft(out_ft, size(x,1),[1,2,3])
end

### 2D FNO structure
mutable struct Net2d{T}
    fc0::Conv
    conv0::SpectralConv{Complex{T},5}
    conv1::SpectralConv{Complex{T},5}
    conv2::SpectralConv{Complex{T},5}
    conv3::SpectralConv{Complex{T},5}
    w0::Conv
    w1::Conv
    w2::Conv
    w3::Conv
    bn0::BatchNorm
    bn1::BatchNorm
    bn2::BatchNorm
    bn3::BatchNorm
    fc1::Conv
    fc2::Conv
end

@Flux.functor Net2d

### 2D FNO constructor
function Net2d(modes::Vector{Int64}, width::Int64; in_channels::Int64=3, out_channels::Int64=1, mid_channels::Int64=128)
    return Net2d(
        Conv((1, 1), in_channels=>width),
        SpectralConv(width, width, modes),
        SpectralConv(width, width, modes),
        SpectralConv(width, width, modes),
        SpectralConv(width, width, modes),
        Conv((1, 1), width=>width),
        Conv((1, 1), width=>width),
        Conv((1, 1), width=>width),
        Conv((1, 1), width=>width),
        BatchNorm(width, identity; ϵ=1f-5, momentum=1f-1),
        BatchNorm(width, identity; ϵ=1f-5, momentum=1f-1),
        BatchNorm(width, identity; ϵ=1f-5, momentum=1f-1),
        BatchNorm(width, identity; ϵ=1f-5, momentum=1f-1),
        Conv((1, 1), width=>mid_channels),
        Conv((1, 1), mid_channels=>out_channels)
    )
end

Net2d(modes::Int64, width::Int64; in_channels::Int64=3, out_channels::Int64=1, mid_channels::Int64=128) = Net2d([modes,modes], width; in_channels=in_channels, out_channels=out_channels, mid_channels=mid_channels)

### 2D FNO forward evaluation
function (B::Net2d{T})(x::AbstractArray{T,4}) where T
    x = B.fc0(x)
    x1 = B.conv0(x)
    x2 = B.w0(x)
    x = B.bn0(x1+x2)
    x = relu.(x)
    x1 = B.conv1(x)
    x2 = B.w1(x)
    x = B.bn1(x1+x2)
    x = relu.(x)
    x1 = B.conv2(x)
    x2 = B.w2(x)
    x = B.bn2(x1+x2)
    x = relu.(x)
    x1 = B.conv3(x)
    x2 = B.w3(x)
    x = B.bn3(x1+x2)
    x = B.fc1(x)
    x = relu.(x)
    x = B.fc2(x)
    # x = dropdims(x,dims=3)
    return x
end

### 3D FNO structure
mutable struct Net3d{T}
    fc0::Conv
    conv0::SpectralConv{Complex{T},6}
    conv1::SpectralConv{Complex{T},6}
    conv2::SpectralConv{Complex{T},6}
    conv3::SpectralConv{Complex{T},6}
    w0::Conv
    w1::Conv
    w2::Conv
    w3::Conv
    bn0::BatchNorm
    bn1::BatchNorm
    bn2::BatchNorm
    bn3::BatchNorm
    fc1::Conv
    fc2::Conv
end

@Flux.functor Net3d

### 3D FNO Constructor
function Net3d(modes::Vector{Int64}, width::Int64; in_channels::Int64=4, out_channels::Int64=1, mid_channels::Int64=128)
    return Net3d(
        Conv((1, 1, 1), in_channels=>width),
        SpectralConv(width, width, modes),
        SpectralConv(width, width, modes),
        SpectralConv(width, width, modes),
        SpectralConv(width, width, modes),
        Conv((1, 1, 1), width=>width),
        Conv((1, 1, 1), width=>width),
        Conv((1, 1, 1), width=>width),
        Conv((1, 1, 1), width=>width),
        BatchNorm(width, identity; ϵ=1f-5, momentum=1f-1),
        BatchNorm(width, identity; ϵ=1f-5, momentum=1f-1),
        BatchNorm(width, identity; ϵ=1f-5, momentum=1f-1),
        BatchNorm(width, identity; ϵ=1f-5, momentum=1f-1),
        Conv((1, 1, 1), width=>mid_channels),
        Conv((1, 1, 1), mid_channels=>out_channels)
    )
    return block
end

Net3d(modes::Int64, width::Int64; in_channels::Int64=4, out_channels::Int64=1, mid_channels::Int64=128) = Net3d([modes,modes,modes], width; in_channels=in_channels, out_channels=out_channels, mid_channels=mid_channels)

### 3D FNO forward evaluation
function (B::Net3d{T})(x::AbstractArray{T,5}) where T
    x = B.fc0(x)
    x1 = B.conv0(x)
    x2 = B.w0(x)
    x = B.bn0(x1+x2)
    x = relu.(x)
    x1 = B.conv1(x)
    x2 = B.w1(x)
    x = B.bn1(x1+x2)
    x = relu.(x)
    x1 = B.conv2(x)
    x2 = B.w2(x)
    x = B.bn2(x1+x2)
    x = relu.(x)
    x1 = B.conv3(x)
    x2 = B.w3(x)
    x = B.bn3(x1+x2)
    x = B.fc1(x)
    x = relu.(x)
    x = B.fc2(x)
    x = dropdims(x,dims=4)
    return x
end