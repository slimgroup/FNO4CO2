function upsample(x)
    ratio = (2, 2, 1, 1)
    (h, w, c, n) = size(x)
    y = ones(Float32, 1, ratio[1], 1, ratio[2], 1, 1) |> gpu
    z = reshape(x, (h, 1, w, 1, c, n)) .* y
    reshape(permutedims(z, (2, 1, 4, 3, 5, 6)), size(x) .* ratio)
end

struct DeepPriorNet

    upsample

    conv1_1
    bn1_1
    conv1_2
    bn1_2

    conv2_1
    bn2_1
    conv2_2
    bn2_2

    conv3_1
    bn3_1
    conv3_2
    bn3_2

    conv_skip
    bn_skip

    bn_cat

    conv4_1
    bn4_1
    conv4_2
    bn4_2

    bn_up1

    conv5_1
    bn5_1
    conv5_2
    bn5_2

    bn_up2

    conv6_1
    bn6_1
    conv6_2
    bn6_2

    conv7

end

@Flux.functor DeepPriorNet

function DeepPriorNet(; num_channel=[16, 32, 64], scale=1.0f0)

    layers = DeepPriorNet(

        x -> upsample(x),

        Conv((5, 5), 3=>num_channel[1], identity, stride=(2, 2), pad=(2, 2)),
        BatchNorm(num_channel[1], identity; ϵ=1.0f-5, momentum=.1f0),
        Conv((5, 5), num_channel[1]=>num_channel[1], identity, stride=(1, 1), pad=(2, 2)),
        BatchNorm(num_channel[1], identity; ϵ=1.0f-5, momentum=.1f0),

        Conv((5, 5), num_channel[1]=>num_channel[2], identity, stride=(2, 2), pad=(2, 2)),
        BatchNorm(num_channel[2], identity; ϵ=1.0f-5, momentum=.1f0),
        Conv((5, 5), num_channel[2]=>num_channel[2], identity, stride=(1, 1), pad=(2, 2)),
        BatchNorm(num_channel[2], identity; ϵ=1.0f-5, momentum=.1f0),

        Conv((5, 5), num_channel[2]=>num_channel[3], identity, stride=(2, 2), pad=(2, 2)),
        BatchNorm(num_channel[3], identity; ϵ=1.0f-5, momentum=.1f0),
        Conv((5, 5), num_channel[3]=>num_channel[3], identity, stride=(1, 1), pad=(2, 2)),
        BatchNorm(num_channel[3], identity; ϵ=1.0f-5, momentum=.1f0),

        Conv((1, 1), num_channel[2]=>num_channel[3], identity, stride=(1, 1), pad=(0, 0)),
        BatchNorm(num_channel[3], identity; ϵ=1.0f-5, momentum=.1f0),

        BatchNorm(num_channel[3]*2, identity; ϵ=1.0f-5, momentum=.1f0),

        Conv((5, 5), num_channel[3]*2=>num_channel[3], identity, stride=(1, 1), pad=(2, 2)),
        BatchNorm(num_channel[3], identity; ϵ=1.0f-5, momentum=.1f0),
        Conv((1, 1), num_channel[3]=>num_channel[3], identity, stride=(1, 1), pad=(0, 0)),
        BatchNorm(num_channel[3], identity; ϵ=1.0f-5, momentum=.1f0),

        BatchNorm(num_channel[3], identity; ϵ=1.0f-5, momentum=.1f0),

        Conv((5, 5), num_channel[3]=>num_channel[2], identity, stride=(1, 1), pad=(2, 2)),
        BatchNorm(num_channel[2], identity; ϵ=1.0f-5, momentum=.1f0),
        Conv((1, 1), num_channel[2]=>num_channel[2], identity, stride=(1, 1), pad=(0, 0)),
        BatchNorm(num_channel[2], identity; ϵ=1.0f-5, momentum=.1f0),

        BatchNorm(num_channel[2], identity; ϵ=1.0f-5, momentum=.1f0),

        Conv((5, 5), num_channel[2]=>num_channel[1], identity, stride=(1, 1), pad=(2, 2)),
        BatchNorm(num_channel[1], identity; ϵ=1.0f-5, momentum=.1f0),
        Conv((1, 1), num_channel[1]=>num_channel[1], identity, stride=(1, 1), pad=(0, 0)),
        BatchNorm(num_channel[1], identity; ϵ=1.0f-5, momentum=.1f0),

        Conv((1, 1), num_channel[1]=>1, identity, stride=(1, 1), pad=(0, 0)),
        )

    return layers
end

function DeepPriorNet(z, layers::DeepPriorNet)

        # layers = layers |> gpu
        x̂ = leakyrelu.(layers.bn1_1(layers.conv1_1(z)), 0.2f0)
        x̂ = leakyrelu.(layers.bn1_2(layers.conv1_2(x̂)), 0.2f0)

        x̂ = leakyrelu.(layers.bn2_1(layers.conv2_1(x̂)), 0.2f0)
        x̂ = leakyrelu.(layers.bn2_2(layers.conv2_2(x̂)), 0.2f0)

        x̂_deep = leakyrelu.(layers.bn3_1(layers.conv3_1(x̂)), 0.2f0)
        x̂_deep = leakyrelu.(layers.bn3_2(layers.conv3_2(x̂_deep)), 0.2f0)
        x̂_deep = layers.upsample(x̂_deep)

        x̂_skip = leakyrelu.(layers.bn_skip(layers.conv_skip(x̂)), 0.2f0)

        x̂ = layers.bn_cat(cat(x̂_skip, x̂_deep, dims=3))

        x̂ = leakyrelu.(layers.bn4_1(layers.conv4_1(x̂)), 0.2f0)
        x̂ = leakyrelu.(layers.bn4_2(layers.conv4_2(x̂)), 0.2f0)

        x̂ = layers.upsample(x̂)
        x̂ = layers.bn_up1(x̂)

        x̂ = leakyrelu.(layers.bn5_1(layers.conv5_1(x̂)), 0.2f0)
        x̂ = leakyrelu.(layers.bn5_2(layers.conv5_2(x̂)), 0.2f0)

        x̂ = layers.upsample(x̂)
        x̂ = layers.bn_up2(x̂)


        x̂ = leakyrelu.(layers.bn6_1(layers.conv6_1(x̂)), 0.2f0)
        x̂ = leakyrelu.(layers.bn6_2(layers.conv6_2(x̂)), 0.2f0)

        x̂ = layers.conv7(x̂)

    return x̂
end
