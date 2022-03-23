 using JUDI
using Zygote, Flux, JOLI
using Zygote: @adjoint
import Base.*
    
# Fixed source forward modeling
struct ForwardIllum{T1}
    F::T1   # forward modeling operator
    q::judiVector
end

# Fixed source forward modeling w/ illumination: forward mode
(FWD::ForwardIllum)(m::AbstractMatrix{Float32}) = FWD.F(;m=m)* FWD.q
@adjoint function (FWD::ForwardIllum)(m::AbstractMatrix{Float32})
    F = FWD.F(;m=m)
    q = FWD.q
    J = judiJacobian(F, q)
    I = inv(judiIllumination(J))
    dobs = F * q
    return dobs, Δ -> (nothing, reshape(I("u") * adjoint(J) * vec(Float32.(Δ)), size(m)))
end