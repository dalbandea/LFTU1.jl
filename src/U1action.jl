import LFTSampling: action

KernelAbstractions.@kernel function U1plaquette!(plx, U, Nx, Ny, ::Type{BC}) where BC <: AbstractBoundaryCondition

    i1, i2 = @index(Global, NTuple)

    iu1 = mod(i1, Nx) + 1
    iu2 = mod(i2, Ny) + 1
    
    plx[i1,i2] = real(U[i1,i2,1] *
                      U[iu1,i2,2] *
                      conj(U[i1,iu2,1] *
                           U[i1,i2,2]))

    if BC == OpenBC
        if i1 == Nx || i2 == Ny
            plx[i1, i2] = zero(eltype(plx))
        end
    end
    
end

KernelAbstractions.@kernel function U1qtop!(plx, U, Nx, Ny)

    i1, i2 = @index(Global, NTuple)

    iu1 = mod(i1, Nx) + 1
    iu2 = mod(i2, Ny) + 1
    
    plx[i1,i2] = angle(U[i1,i2,1] *
                            U[iu1,i2,2] *
                            conj(U[i1,iu2,1] *
                                 U[i1,i2,2]))
end


function force!(U1ws::U1Quenched, hmcws::AbstractHMC)
    return gauge_force!(U1ws, hmcws)
end

function gauge_force!(U1ws::U1, hmcws::AbstractHMC)
    lp = U1ws.params
    event = U1quenchedforce!(U1ws.device)(hmcws.frc1, hmcws.frc2, U1ws.U, lp.beta, lp.iL[1], lp.iL[2], lp.BC, ndrange=(lp.iL[1], lp.iL[2]), workgroupsize=U1ws.kprm.threads)
    synchronize(U1ws.device)
    return nothing
end

KernelAbstractions.@kernel function U1quenchedforce!(frc1, frc2, U, beta, Nx, Ny, ::Type{BC}) where BC <: AbstractBoundaryCondition
    
    i1, i2 = @index(Global, NTuple)

    iu1 = mod(i1, Nx) + 1
    iu2 = mod(i2, Ny) + 1


    v = beta * imag(U[i1,i2,1] * U[iu1,i2,2] * conj(U[i1,iu2,1] * U[i1,i2,2]))

    if BC == OpenBC
        if i1 == Nx || i2 == Ny
            v = zero(eltype(v))
        end
    end

    frc1[i1,i2,1]  = -v 
    frc1[i1,i2,2]  =  v 
    frc2[iu1,i2,2] = -v 
    frc2[i1,iu2,1] =  v 

end

action(U1ws::U1Quenched, hmcws::AbstractHMC) = gauge_action(U1ws)
action(U1ws::U1Quenched) = gauge_action(U1ws)

gauge_action(U1ws::U1) = U1quenchedaction(U1ws)

function U1quenchedaction(U1ws::U1)
    lp = U1ws.params
    return U1quenchedaction(U1ws.U, lp.beta, lp.iL[1], lp.iL[2], lp.BC, U1ws.device, U1ws.kprm.threads, U1ws.kprm.blocks)
end

function U1quenchedaction(U, beta, Nx, Ny, BC, device, threads, blocks)
    plaquettes = to_device(device, zeros(real(eltype(U)), Nx, Ny))
    return U1quenchedaction(plaquettes, U, beta, Nx, Ny, BC, device, threads, blocks)
end

function U1quenchedaction(plaquettes, U, beta, Nx, Ny, BC, device, threads, blocks)
    event = U1plaquette!(device)(plaquettes, U, Nx, Ny, BC, ndrange=(Nx, Ny), workgroupsize=threads)
    synchronize(device)

    if BC == OpenBC
        Nx = Nx - 1
        Ny = Ny - 1
    end
    S = beta * ( Nx * Ny - reduce(+, plaquettes) )

    return S
end

function top_charge(U1ws::U1)
    lp = U1ws.params
    return U1topcharge(U1ws.U, lp.beta, lp.iL[1], lp.iL[2], U1ws.device, U1ws.kprm.threads, U1ws.kprm.blocks)
end

function U1topcharge(U, beta, Nx, Ny, device, threads, blocks)
    plaquettes = to_device(device, zeros(Float64, Nx, Ny))
    return U1topcharge(plaquettes, U, beta, Nx, Ny, device, threads, blocks)
end

function U1topcharge(plaquettes, U, beta, Nx, Ny, device, threads, blocks)
    event = U1qtop!(device)(plaquettes, U, Nx, Ny, ndrange=(Nx, Ny), workgroupsize=threads)
    synchronize(device)
    Q = reduce(+, plaquettes) / (2.0*pi)
    return Q
end

function top_charge_density(U1ws::U1)
    lp = U1ws.params
    plaquettes = to_device(U1ws.device, zeros(Float64, lp.iL[1], lp.iL[2]))
    top_charge_density!(plaquettes, U1ws)
    return plaquettes
end

function top_charge_density!(plaquettes, U1ws::U1)
    lp = U1ws.params
    return top_charge_density!(plaquettes, U1ws.U, lp.beta, lp.iL[1], lp.iL[2], U1ws.device, U1ws.kprm.threads, U1ws.kprm.blocks)
end

function top_charge_density!(plaquettes, U, beta, Nx, Ny, device, threads, blocks)
    event = U1qtop!(device)(plaquettes, U, Nx, Ny, ndrange=(Nx, Ny), workgroupsize=threads)
    synchronize(device)
    return nothing
end
export top_charge_density, top_charge_density!


# ======================= #
# ====== U1 Nf = 2 ====== #
# ======================= #


action(U1ws::U1Nf2, hmcws::AbstractHMC) = gauge_action(U1ws) + pfaction(U1ws, hmcws)

function pfaction(U1ws::U1Nf2, hmcws::AbstractHMC)
    invert!(hmcws.X, gamm5Dw_sqr_msq!, hmcws.F, U1ws.sws, U1ws)
    return real(LinearAlgebra.dot(hmcws.X, hmcws.F))
end

function pf_force!(U1ws::Union{U1Nf2,U1Nf}, hmcws::AbstractHMC)
    lp = U1ws.params
    event = U1_tr_dQwdU!(U1ws.device)(hmcws.pfrc, U1ws.U, hmcws.X, hmcws.g5DX,
                                    lp.iL[1], lp.iL[2], 
                                    ndrange=(lp.iL[1], lp.iL[2]),
                                    workgroupsize=U1ws.kprm.threads)
    synchronize(U1ws.device)
    return nothing
end

function force!(U1ws::U1Nf2, hmcws::AbstractHMC)
	# Solve DX = F for X
    iter = invert!(hmcws.X, gamm5Dw_sqr_msq!, hmcws.F, U1ws.sws, U1ws)

	# Apply gamm5D to X
    gamm5Dw!(hmcws.g5DX, hmcws.X, U1ws)
	
	# Get fermion part of the force in U1ws.pfrc
    pf_force!(U1ws, hmcws)

	# Get gauge part of the force in U1ws.frc1 and U1ws.frc2
    gauge_force!(U1ws, hmcws)

    return nothing
end


# ======================= #
# ======== U1 Nf ======== #
# ======================= #

action(U1ws::U1Nf, hmcws::AbstractHMC) = gauge_action(U1ws) + pfaction(U1ws, hmcws)

function pfaction(U1ws::U1Nf, hmcws::AbstractHMC)
    S_pf = zero(U1ws.PRC)
    for j in 1:length(U1ws.params.am0)
        LFTU1.MultiCG(hmcws.X, hmcws.F[j], U1ws.params.am0[j], U1ws.rprm[j], U1ws)
        S_pf += LinearAlgebra.dot(hmcws.X, hmcws.F[j]) |> real
    end
    return S_pf
end

function force!(U1ws::U1Nf, hmcws::AbstractHMC)
    am0 = U1ws.params.am0
    N_fermions = length(am0)
    rprm = U1ws.rprm
    aux = similar(hmcws.pfrc)                 # real, as the force
    aux .= zero(eltype(aux))

    for j in 1:N_fermions
        for i in 1:rprm[j].n
            # X = (D†D+μ²)⁻¹ϕⱼ
            iter = invert!(hmcws.X, gamm5Dw_sqr_musq_am0_mu!(am0[j], rprm[j].mu[i]), hmcws.F[j], U1ws.sws, U1ws)

            # Apply gamm5D to X
            gamm5Dw!(hmcws.g5DX, hmcws.X, am0[j], U1ws)

            # Get fermion part of the force in hmcws.pfrc
            pf_force!(U1ws, hmcws)

            # Dump it to aux
            aux .= aux .+ rprm[j].rho[i]*hmcws.pfrc
        end
    end

    # Dump aux to pfrc
    hmcws.pfrc .= aux

	# Get gauge part of the force in U1ws.frc1 and U1ws.frc2
    gauge_force!(U1ws, hmcws)

    return nothing
end
