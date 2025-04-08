import LFTSampling: sample!, generate_momenta!, Hamiltonian, update_momenta!, update_fields!
import LFTSampling: flip_momenta_sign!

# ======================= #
# ===== General U1 ====== #
# ======================= #

KernelAbstractions.@kernel function U1generate_momenta!(mom, Nx, Ny, ::Type{BC}) where BC <: AbstractBoundaryCondition

    i1, i2 = @index(Global, NTuple)

    mom[i1, i2, 1] = randn(eltype(mom))
    mom[i1, i2, 2] = randn(eltype(mom))

    if BC == OpenBC
        if i1 == Nx
            mom[i1, i2, 1] = zero(eltype(mom))
        end
        if i2 == Ny
            mom[i1, i2, 2] = zero(eltype(mom))
        end
    end
end

function generate_momenta!(U1ws::U1, hmcws::AbstractHMC)
    lp = U1ws.params
    event = U1generate_momenta!(U1ws.device)(hmcws.mom, lp.iL[1], lp.iL[2], lp.BC, ndrange=(lp.iL[1], lp.iL[2]), workgroupsize=U1ws.kprm.threads)
    synchronize(U1ws.device)
    return nothing
end

# function generate_momenta!(U1ws::U1, hmcws::AbstractHMC)
#     # Create momenta for U1
#     hmcws.mom .= to_device(U1ws.device, randn(U1ws.PRC, size(hmcws.mom)))
#     return nothing
# end

function Hamiltonian(U1ws::U1, hmcws::AbstractHMC)
    H = mapreduce(x -> x^2, +, hmcws.mom)/2.0 + action(U1ws, hmcws)
    return H
end

function update_fields!(U1ws::T, epsilon, hmcws::AbstractHMC) where T <: U1
    lp = U1ws.params
    event = U1_update_field!(U1ws.device)(U1ws.U, hmcws.mom, epsilon, ndrange=(lp.iL[1], lp.iL[2]), workgroupsize=U1ws.kprm.threads)
    synchronize(U1ws.device)
    return nothing
end

KernelAbstractions.@kernel function U1_update_field!(U, mom, epsilon)

    i1, i2 = @index(Global, NTuple)

    for id in 1:2
        U[i1,i2,id] = complex(cos(epsilon*mom[i1,i2,id]), sin(epsilon*mom[i1,i2,id])) * U[i1,i2,id]
    end
end


function winding_step!(Lw::Int64, U1ws::U1, samplerws::AbstractHMC)
    ws_cp = deepcopy(U1ws)
    
    generate_pseudofermions!(U1ws, samplerws)

    sini = action(U1ws, samplerws)

    r = rand()

    if r > 0.5
        winding!(Lw, U1ws, sign = 1)
    else
        winding!(Lw, U1ws, sign = -1)
    end

    sfin = action(U1ws, samplerws)

    ds = sfin - sini

    LFTSampling.metropolis_accept_reject!(U1ws, ws_cp, ds)

    return nothing
end


function sample!(U1ws::U1, samplerws::AbstractHMC; N_windings = 0, Lw = 0)
    @info("- HMC step")
    hmc!(U1ws, samplerws)
    @info("- $N_windings winding steps")
    for i in 1:N_windings
        winding_step!(Lw, U1ws, samplerws)
    end
    return nothing
end

# ======================= #
# ===== U1 Quenched ===== #
# ======================= #

function flip_momenta_sign!(hmcws::U1quenchedHMC)
    hmcws.mom .= .- hmcws.mom
    return nothing
end

function update_momenta!(U1ws::U1Quenched, epsilon, hmcws::AbstractHMC)
    force!(U1ws, hmcws)
    hmcws.mom .= hmcws.mom .+ epsilon .* (hmcws.frc1 .+ hmcws.frc2)
    return nothing
end


# ======================= #
# ====== U1 Nf = 2 ====== #
# ======================= #

import LFTSampling: generate_pseudofermions!

function flip_momenta_sign!(hmcws::U1Nf2HMC)
    hmcws.mom .= .- hmcws.mom
    return nothing
end

function generate_pseudofermions!(U1ws::U1Nf2, hmcws::AbstractHMC)
    lp = U1ws.params

    hmcws.X .= to_device(U1ws.device, randn(complex(U1ws.PRC), lp.iL[1], lp.iL[2], 2))
    gamm5Dw!(hmcws.F, hmcws.X, U1ws)
    hmcws.g5DX .= to_device(U1ws.device, zeros(complex(U1ws.PRC), lp.iL[1], lp.iL[2], 2))
    return nothing
end

function update_momenta!(U1ws::Union{U1Nf2,U1Nf}, epsilon, hmcws::AbstractHMC)
    # Compute force
    force!(U1ws, hmcws)

	# Final force is frc1+frc2+frc
    hmcws.mom .= hmcws.mom .+ epsilon .* (hmcws.frc1 .+ hmcws.frc2 .+ hmcws.pfrc)

	return nothing
end


# ======================= #
# ======== U1 Nf ======== #
# ======================= #


function flip_momenta_sign!(hmcws::U1NfHMC)
    hmcws.mom .= .- hmcws.mom
    return nothing
end


function generate_pseudofermions!(U1ws::U1Nf, hmcws::AbstractHMC)
    lp = U1ws.params
    N_fermions = length(U1ws.params.am0)

    hmcws.g5DX .= to_device(U1ws.device, zeros(complex(U1ws.PRC), lp.iL[1], lp.iL[2], 2))

    hpf_ini = zero(U1ws.PRC)

    # Fill array of N_fermions pseudofermion fields and complete initial Hamiltonian
    for j in 1:N_fermions
        hmcws.X .= to_device(U1ws.device, randn(complex(U1ws.PRC), lp.iL[1], lp.iL[2], 2))
        hpf_ini += mapreduce(x -> abs2(x), +, hmcws.X)
        generate_pseudofermion!(hmcws.F[j], U1ws.params.am0[j], U1ws.rprm[j], U1ws, hmcws)
    end
    return hpf_ini
end


"""
    generate_pseudofermion(F, am0, rprm::RHMCParm, U1ws::U1Nf, hmcws::AbstractHMC)

Obtain pseudofermion from a random field `X` given the RHMC parameters `rprm`.
Equivalent to ``A_{k,l}`` operator in Lüscher eq. (3.8).
"""
function generate_pseudofermion!(F, am0, rprm::RHMCParm, U1ws::U1Nf, hmcws::AbstractHMC)
    # S_pf =  ϕ† ∏( (D†D + μᵢ²)⁻¹ (D†D + νᵢ²) ) ϕ = X† X
    # ϕ = ∏( (γD + iνᵢ)⁻¹(γD + iμᵢ) ) X  if X is random normal.
    F .= hmcws.X
    for i in 1:rprm.n
        # ϕⱼ = (γD+iμ)X = (γD+iμ)ϕⱼ
        aux_F = copy(F)
        gamm5Dw!(F, aux_F, am0, U1ws)
        F .= F .+ im*rprm.mu[i] .* aux_F
        # ϕⱼ = (D†D+ν²)⁻¹ϕⱼ
        aux_F .= F
        # CG(F, U, aux_F, am0, rprm.nu[i], CGmaxiter, CGtol, gamm5Dw_sqr_musq, prm, kprm)
        iter = invert!(F, gamm5Dw_sqr_musq_am0_mu!(am0, rprm.nu[i]), aux_F, U1ws.sws, U1ws)
        # ϕⱼ = (γD-iν)ϕⱼ
        aux_F .= F
        gamm5Dw!(F, aux_F, am0, U1ws)
        F .= F .- im*rprm.nu[i] .* aux_F
    end
end


"""
    MultiCG(so, U, si, am0, maxiter, eps, A, rprm, prm, kprm)

Applies the partial fraction decomposition of  rational approximation ``R(A) =
A^{-1/2}`` without constant factors (as opposed to function `R`) to the field
`si`, storing it in `so`, with `A` an operator. Used for pseudofermion-field
generation, see Luscher eq. (3.9)
"""
function MultiCG(so, si, am0, rprm::RHMCParm, U1ws::U1)
	aux = similar(so)
	so .= si  # this accounts for identity in partial fraction descomposition
	for j in 1:rprm.n
        iter = invert!(aux, gamm5Dw_sqr_musq_am0_mu!(am0, rprm.mu[j]), si, U1ws.sws, U1ws)
		so .= so .+ rprm.rho[j]*aux
	end

    return nothing
end
