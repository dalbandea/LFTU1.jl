import LFTSampling: copy!, sampler, AbstractSolver

# ======================= #
# ===== General U1 ====== #
# ======================= #

function copy!(U1ws_dst::U1, U1ws_src::U1)
    U1ws_dst.U .= U1ws_src.U
    return nothing
end

function randomize!(U1ws::U1)
    U1ws.U .= to_device(U1ws.device, exp.(im * Random.rand(U1ws.PRC, size(U1ws.U)) * 2 * pi))
    return nothing
end

function coldstart!(U1ws::U1)
    U1ws.U .= one(complex(U1ws.PRC))
    return nothing
end

function random_gauge_trafo(U1ws::U1)
    lp = U1ws.params

    Omega = exp.(im*2pi*Random.rand(Float64, lp.iL[1], lp.iL[2]))

    for i in 1:lp.iL[1], j in 1:lp.iL[2]
        iu = mod1(i+1, lp.iL[1])
        ju = mod1(j+1, lp.iL[2])
        U1ws.U[i,j,1] = Omega[i,j] * U1ws.U[i,j,1] * conj(Omega[iu,j])
        U1ws.U[i,j,2] = Omega[i,j] * U1ws.U[i,j,2] * conj(Omega[i,ju])
    end

    return nothing
end

function winding(Lw, U1ws::U1; sign = 1)
    sign == 1 || sign == -1 || Base.error("Sign of winding must be +1 or -1.")

    # pos = Int64(floor(lsize*lsize*rand()))
    pos = 1

    x1  = Int64(floor((pos-1)/lsize)) +1
    x2  = mod1(pos, lsize)

    phase = zeros(ComplexF64, Lw+1, Lw+1)

    for i in 1:Lw+1
        phase[1, i] = (i-1)*pi/(2*Lw)
        phase[Lw+1, i] = 3*pi/2 - (i-1)*pi/(2*Lw);
        phase[i, 1] = 2*pi - (i-1)*pi/(2*Lw);
        phase[i, Lw+1] = pi/2 + (i-1)*pi/(2*Lw);
    end
    phase[1, 1] = 2*pi;

    Omega = exp.(sign*im*phase)

    for i in 1:Lw, j in 1:Lw
        U1ws.U[i,j,1] = Omega[i,j] * U1ws.U[i,j,1] * conj(Omega[i+1,j])
        U1ws.U[i,j,2] = Omega[i,j] * U1ws.U[i,j,2] * conj(Omega[i,j+1])
    end
    for i in 1:Lw
        U1ws.U[i,Lw+1,1] = Omega[i,Lw+1] * U1ws.U[i,Lw+1,1] * conj(Omega[i+1,Lw+1])
        U1ws.U[Lw+1,i,2] = Omega[Lw+1,i] * U1ws.U[Lw+1,i,2] * conj(Omega[Lw+1,i+1])
    end

    return nothing
end

# ======================= #
# ===== U1 Quenched ===== #
# ======================= #

@doc raw"""
    struct U1workspace{T}

Allocates all the necessary fields for a HMC simulation of a U(1) model:

- `PRC`: precision; must be `ComplexF64`.
- `U`: ``U`` gauge field.
- `frc`
"""
struct U1quenchedworkspace{T1, A <: AbstractArray} <: U1Quenched
    PRC::Type{T1}
    U::A
    params::U1QuenchedParm
    device#::Union{KernelAbstractions.Device, ROCKernels.ROCDevice}
    kprm::KernelParm
end

function (s::Type{U1Quenched})(::Type{T1}, ::Type{T2} = complex(T1); custom_init = nothing, device = KernelAbstractions.CPU(), kwargs...) where {T1, T2}
    lp = U1QuenchedParm(;kwargs...)
    return U1quenchedworkspace(T1, T2, lp, device, KernelParm(lp); custom_init = custom_init)
end

function U1quenchedworkspace(::Type{T1}, ::Type{T2}, lp::U1Parm, device, kprm; custom_init = nothing) where {T1,T2}
    if custom_init == nothing
        U = to_device(device, ones(T2, lp.iL..., 2))
    else
        U = custom_init
    end
    return U1quenchedworkspace{T1, typeof(U)}(T1, U, lp, KernelAbstractions.get_backend(U), kprm)
end

struct U1quenchedHMC{A <: AbstractArray} <: AbstractHMC
    params::HMC
    frc1::A
    frc2::A
    mom::A
end


function U1quenchedHMC(u1ws::U1Quenched, hmcp::HMCParams)
    frc1 = to_device(u1ws.device, zeros(u1ws.PRC, u1ws.params.iL..., 2))
    frc2 = similar(frc1)
    mom = similar(frc1)
    return U1quenchedHMC(hmcp, frc1, frc2, mom)
end

sampler(lftws::U1Quenched, hmcp::HMCParams) = U1quenchedHMC(lftws, hmcp)

# ======================= #
# ====== U1 Nf = 2 ====== #
# ======================= #

struct U1Nf2workspace{T, A <: AbstractArray, S <: AbstractSolver} <: U1Nf2
    PRC::Type{T}
    U::A
    params::U1Nf2Parm
    device#::Union{KernelAbstractions.Device, ROCKernels.ROCDevice}
    kprm::KernelParm
    sws::S
end

function U1Nf2workspace(::Type{T1}, ::Type{T2}, lp::U1Nf2Parm, device, kprm, maxiter::Int64 = 10000,
        tol::Float64 = 1e-14; custom_init = nothing) where {T1, T2}
    if custom_init == nothing
        U = to_device(device, ones(T2, lp.iL..., 2))
    else
        U = custom_init
    end
    sws = CG(maxiter, tol, U)
    return U1Nf2workspace{T1, typeof(U), typeof(sws)}(T1, U, lp, KernelAbstractions.get_backend(U), kprm, sws)
end

function (s::Type{U1Nf2})(::Type{T1}, ::Type{T2} = complex(T1); custom_init = nothing, device = KernelAbstractions.CPU(), maxiter::Int64 = 10000, tol::Float64 = 1e-14, kwargs...) where {T1, T2}
    lp = U1Nf2Parm(;kwargs...)
    return U1Nf2workspace(T1, T2, lp, device, KernelParm(lp), maxiter, tol, custom_init = custom_init)
end

struct U1Nf2HMC{A1 <: AbstractArray, A2 <: AbstractArray} <: AbstractHMC
    params::HMC
    X::A1
    F::A1
    g5DX::A1
    frc1::A2 # gauge force
    frc2::A2 # gauge force
    pfrc::A2 # pf force
    mom::A2
end

function U1Nf2HMC(u1ws::U1Nf2, hmcp::HMCParams)
    X = similar(u1ws.U)
    F = similar(u1ws.U)
    g5DX = similar(u1ws.U)
    frc1 = to_device(u1ws.device, zeros(u1ws.PRC, u1ws.params.iL..., 2))
    frc2 = similar(frc1)
    pfrc = similar(frc1)
    mom = similar(frc1)
    return U1Nf2HMC{typeof(X), typeof(frc1)}(hmcp, X, F, g5DX, frc1, frc2, pfrc, mom)
end

sampler(lftws::U1Nf2, hmcp::HMCParams) = U1Nf2HMC(lftws, hmcp)


# ======================= #
# ======== U1 Nf ======== #
# ======================= #


struct U1Nfworkspace{T, A <: AbstractArray, S <: AbstractSolver} <: U1Nf
    PRC::Type{T}
    U::A
    params::U1NfParm
    device
    kprm::KernelParm
    rprm::Array{RHMCParm}
    sws::S
end

function U1Nfworkspace(::Type{T1}, ::Type{T2}, lp::U1NfParm, device, kprm, rprm, maxiter::Int64 = 10000,
        tol::Float64 = 1e-14; custom_init = nothing) where {T1, T2}
    if custom_init == nothing
        U = to_device(device, ones(T2, lp.iL..., 2))
    else
        U = custom_init
    end
    sws = CG(maxiter, tol, U)
    return U1Nfworkspace{T1, typeof(U), typeof(sws)}(T1, U, lp, KernelAbstractions.get_backend(U), kprm, rprm, sws)
end

function (s::Type{U1Nf})(::Type{T1}, ::Type{T2} = complex(T1); ns_rat = nothing,
                         r_as = nothing, r_bs = nothing, custom_init = nothing,
                         device = KernelAbstractions.CPU(), maxiter::Int64 =
                         10000, tol::Float64 = 1e-14, kwargs...) where {T1, T2}
    lp = U1NfParm(;kwargs...)
    
    if ns_rat == nothing
        N_fermions = length((;kwargs...).am0)
        ns_rat = fill(1, N_fermions)
        r_as = copy((;kwargs...).am0)
        r_bs = fill(6.0, N_fermions)
    end
    rprm = get_rhmc_params(ns_rat, r_as, r_bs)
    return U1Nfworkspace(T1, T2, lp, device, KernelParm(lp), rprm, maxiter, tol, custom_init = custom_init)
end


struct U1NfHMC{A1 <: AbstractArray, A2 <: AbstractArray} <: AbstractHMC
    params::HMC
    X::A1
    F::Array{A1}
    g5DX::A1
    frc1::A2 # gauge force
    frc2::A2 # gauge force
    pfrc::A2 # pf force
    mom::A2
end

function U1NfHMC(u1ws::U1Nf, hmcp::HMCParams)
    N_fermions = length(u1ws.params.am0)
    X = similar(u1ws.U)
    F = Array{typeof(u1ws.U)}(undef, N_fermions)
    for i in 1:N_fermions
        F[i] = similar(u1ws.U)
    end
    tmpF = similar(u1ws.U)
    g5DX = similar(u1ws.U)
    frc1 = to_device(u1ws.device, zeros(u1ws.PRC, u1ws.params.iL..., 2))
    frc2 = similar(frc1)
    pfrc = similar(frc1)
    mom = similar(frc1)
    return U1NfHMC{typeof(X), typeof(frc1)}(hmcp, X, F, g5DX, frc1, frc2, pfrc, mom)
end

sampler(lftws::U1Nf, hmcp::HMCParams) = U1NfHMC(lftws, hmcp)
