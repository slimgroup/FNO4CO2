export Net2d

mutable struct SpectralConv2d{T,N}
    weights1::AbstractArray{T,N}
    weights2::AbstractArray{T,N}
end

@Flux.functor SpectralConv2d

# Constructor
function SpectralConv2d(in_channels::Integer, out_channels::Integer, modes1::Integer, modes2::Integer)
    scale = (1f0 / (in_channels * out_channels))
    if gpu_flag
        weights1 = scale*randn(Complex{Float32}, modes1, modes2, in_channels, out_channels) |> gpu
        weights2 = scale*randn(Complex{Float32}, modes1, modes2, in_channels, out_channels) |> gpu
    else
        weights1 = scale*randn(Complex{Float32}, modes1, modes2, in_channels, out_channels)
        weights2 = scale*randn(Complex{Float32}, modes1, modes2, in_channels, out_channels)
    end
    return SpectralConv2d{Complex{Float32}, 4}(weights1, weights2)
end

function compl_mul2d(x::AbstractArray{Complex{Float32}}, y::AbstractArray{Complex{Float32}})
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

function (L::SpectralConv2d)(x::AbstractArray{Float32})
    # x in (size_x, size_y, channels, batchsize)
    x_ft = rfft(x,[1,2])
    modes1 = size(L.weights1,1)
    modes2 = size(L.weights1,2)
    out_ft = cat(cat(compl_mul2d(x_ft[1:modes1, 1:modes2,:,:], L.weights1),
                zeros(ComplexF32, modes1, size(x_ft,2)-2*modes2, size(x_ft,3), size(x_ft,4)),
                compl_mul2d(x_ft[1:modes1, end-modes2+1:end,:,:], L.weights2),dims=2),
                zeros(ComplexF32, size(x_ft,1)-modes1, size(x_ft,2), size(x_ft,3), size(x_ft,4)),dims=1)
    x = irfft(out_ft, size(x,1),[1,2])
end

mutable struct SimpleBlock2d
    fc0::Conv
    conv0::SpectralConv2d
    conv1::SpectralConv2d
    conv2::SpectralConv2d
    conv3::SpectralConv2d
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

@Flux.functor SimpleBlock2d

function SimpleBlock2d(modes1::Integer, modes2::Integer, width::Integer, in_channels::Integer, out_channels::Integer, mid_channels::Integer)
    block = SimpleBlock2d(
        Conv((1, 1), in_channels=>width),
        SpectralConv2d(width, width, modes1, modes2),
        SpectralConv2d(width, width, modes1, modes2),
        SpectralConv2d(width, width, modes1, modes2),
        SpectralConv2d(width, width, modes1, modes2),
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
    return block
end

function (B::SimpleBlock2d)(x::AbstractArray{Float32})
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
    return x
end

mutable struct Net2d
    conv1::SimpleBlock2d
end

@Flux.functor Net2d

function Net2d(modes::Integer, width::Integer; in_channels::Integer=3, out_channels::Integer=1, mid_channels::Integer=128)
    return Net2d(SimpleBlock2d(modes,modes,width,in_channels,out_channels,mid_channels))
end

function (NN::Net2d)(x::AbstractArray{Float32})
    x = NN.conv1(x)
    x = dropdims(x,dims=3)
end
