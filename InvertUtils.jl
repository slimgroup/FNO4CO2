using Zygote: @adjoint
struct InvertNetFwd{T1}
    G::T1
end
struct InvertNetRev{T1}
    G::T1
end
(G::InvertNetFwd)(x::AbstractArray{T, 4}) where T = G.G.logdet ? G.G.forward(x)[1] : G.G.forward(x)
(G::InvertNetRev)(z::Vector{T}) where T = G.G.inverse(z)
@adjoint function(G::InvertNetFwd)(x::AbstractArray{T, 4}) where T
    z = G.G.logdet ? G.G.forward(x)[1] : G.G.forward(x)
    return z, Δ -> (nothing, G.G.backward(vec(Δ),vec(z))[1])
end
@adjoint function(G::InvertNetRev)(z::Vector{T}) where T
    x = G.G.inverse(z)
    return x, Δ -> (nothing, reverse(G.G).backward(Δ,x)[1])
end