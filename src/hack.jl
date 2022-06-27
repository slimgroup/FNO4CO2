export InvertNetRev

using Zygote: @adjoint
struct InvertNetRev{T1}
    G::T1
end
(G::InvertNetRev)(z::Vector{T}) where T = G.G.inverse(z)
@adjoint function(G::InvertNetRev)(z::Vector{T}) where T
    x = G.G.inverse(z)
    return x, Δ -> (nothing, reverse(G.G).backward(Δ,x)[1])
end